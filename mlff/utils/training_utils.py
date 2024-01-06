import jraph
import jax
import jax.numpy as jnp
from mlff.masking.mask import safe_mask
import numpy as np
import optax
from orbax import checkpoint
from pathlib import Path
from typing import Callable, Dict
import wandb


property_to_mask = {
    'energy': 'graph_mask',
    'stress': 'graph_mask',
    'forces': 'node_mask',
}


def scaled_safe_masked_mse_loss(y, y_true, scale, msk):
    """

    Args:
        y (): shape: (B,d1, *, dN)
        y_true (): (B,d1, *, dN)
        scale (): (d1, *, dN) or everything broadcast-able to (B, d1, *, dN)
        msk (): shape: (B)

    Returns:

    """
    full_mask = ~jnp.isnan(y_true) & jnp.expand_dims(msk, [y_true.ndim - 1 - o for o in range(0, y_true.ndim - 1)])
    v = safe_mask(full_mask, fn=lambda u: scale * (u - y)**2, operand=y_true)
    den = full_mask.reshape(-1).sum().astype(dtype=v.dtype)
    return safe_mask(den > 0, lambda x: v.reshape(-1).sum() / x, den, 0)


def make_loss_fn(obs_fn: Callable, weights: Dict, scales: Dict = None):
    # Targets are collected based on the loss weights.
    targets = list(weights.keys())

    if scales is None:
        _scales = {k: jnp.ones(1) for k in targets}
    else:
        _scales = scales

    @jax.jit
    def loss_fn(params, batch: Dict[str, jnp.ndarray]):
        # Everything that is not a target is a input.
        inputs = {k: v for k, v in batch.items() if k not in targets}

        # Collect the targets.
        outputs_true = {k: v for k, v in batch.items() if k in targets}

        # Make predictions.
        outputs_predict = obs_fn(params, **inputs)
        loss = jnp.zeros(1)
        metrics = {}
        # Iterate over the targets, calculate loss and multiply with loss weights and scales.
        for target in targets:
            _l = scaled_safe_masked_mse_loss(y=outputs_predict[target],
                                             y_true=outputs_true[target],
                                             scale=_scales[target],
                                             msk=inputs[property_to_mask[target]]
                                             )

            loss += weights[target] * _l
            metrics.update({target: _l / _scales[target].mean()})

        loss = jnp.reshape(loss, ())
        metrics.update({'loss': loss})

        return loss, metrics

    return loss_fn


def make_training_step_fn(optimizer, loss_fn):

    @jax.jit
    def training_step_fn(params, opt_state, batch):
        """
        Training step.

        Args:
            params (FrozenDict): Parameter dictionary.
            opt_state: Optax optimizer state.
            batch (Tuple): Batch of validation data.

        Returns:
            Updated state and metrics.

        """
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch)
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params=params, updates=updates)
        metrics['gradients_norm'] = optax.global_norm(grads)
        return params, metrics
    return training_step_fn


def make_validation_step_fn(metric_fn):

    @jax.jit
    def validation_step_fn(params, batch) -> Dict[str, jnp.ndarray]:
        """
        Validation step.

        Args:
            params (FrozenDict): Parameters.
            batch (Tuple): Batch of validation data.

        Returns:
            Validation metrics.
        """
        _, metrics = metric_fn(params, batch)
        return metrics

    return validation_step_fn


def fit(
        model,
        optimizer,
        loss_fn,
        graph_to_batch_fn,
        training_data,
        validation_data,
        batch_max_num_nodes,
        batch_max_num_edges,
        batch_max_num_graphs,
        num_epochs: int = 100,
        ckpt_dir: str = None,
        ckpt_manager_options: dict = None,
        eval_every_num_steps: int = 1000,
        allow_restart=False,
        training_seed: int = 0,
        model_seed: int = 0,
        use_wandb: bool = True,
):
    """
    Fit model.

    Args:
        model: flax module.
        optimizer: optax optimizer.
        loss_fn (Callable): The loss function. Gradient is computed wrt to this function.
        graph_to_batch_fn (Callable): Function that takes a batched graph and returns a batch for the loss_fn.
        training_data (Sequence): Sequence of jraph.GraphTuples.
        validation_data (Sequence): Sequence of jraph.GraphTuples.
        batch_max_num_nodes (int): Maximal number of nodes per batch.
        batch_max_num_edges (int): Maximal number of edges per batch.
        batch_max_num_graphs (int): Maximal number of graphs per batch.
        num_epochs (int): Number of training epochs.
        ckpt_dir (str): Checkpoint path.
        ckpt_manager_options (dict): Checkpoint manager options.
        eval_every_num_steps (int): Evaluate the metrics every num-th step
        allow_restart: Restarts from existing checkpoints are allowed.
        training_seed (int): Random seed for shuffling of training data.
        model_seed (int): Random seed for model initialization.
        use_wandb (bool): Log statistics to WeightsAndBias. If true, wandb.init() must be called before call to fit().
    Returns:

    """
    numpy_rng = np.random.RandomState(seed=training_seed)
    jax_rng = jax.random.PRNGKey(seed=model_seed)

    # Create checkpoint directory.
    ckpt_dir = Path(ckpt_dir).resolve().absolute()
    ckpt_dir.mkdir(exist_ok=True)

    # Create orbax CheckpointManager.
    if ckpt_manager_options is None:
        ckpt_manager_options = {'max_to_keep': 1}

    options = checkpoint.CheckpointManagerOptions(
        best_fn=lambda u: u['loss'],
        best_mode='min',
        step_prefix='ckpt',
        **ckpt_manager_options
    )

    ckpt_mngr = checkpoint.CheckpointManager(
        ckpt_dir,
        {'params': checkpoint.AsyncCheckpointer(checkpoint.PyTreeCheckpointHandler())},
        options=options
    )

    training_step_fn = make_training_step_fn(optimizer, loss_fn)
    validation_step_fn = make_validation_step_fn(loss_fn)

    processed_graphs = 0
    processed_nodes = 0
    step = 0

    params = None
    opt_state = None
    for epoch in range(num_epochs):
        # Shuffle the training data.
        numpy_rng.shuffle(training_data)

        # Create batched graphs from list of graphs.
        iterator_training = jraph.dynamically_batch(
            training_data,
            n_node=batch_max_num_nodes,
            n_edge=batch_max_num_edges,
            n_graph=batch_max_num_graphs
        )

        # Start iteration over batched graphs.
        for graph_batch_training in iterator_training:
            batch_training = graph_to_batch_fn(graph_batch_training)
            processed_graphs += batch_training['num_of_non_padded_graphs']
            processed_nodes += batch_max_num_nodes - jraph.get_number_of_padding_with_graphs_nodes(graph_batch_training)
            batch_training = jax.tree_map(jnp.array, batch_training)

            # In the first step, initialize the parameters or load from existing checkpoint.
            if step == 0:
                # Check if checkpoint already exists.
                latest_step = ckpt_mngr.latest_step()
                if latest_step is not None:
                    if allow_restart:
                        params = ckpt_mngr.restore(
                            latest_step,
                            {'params': checkpoint.AsyncCheckpointer(checkpoint.PyTreeCheckpointHandler())}
                        )['params']
                        step += latest_step
                        print(f'Re-start training from {latest_step}.')
                    else:
                        raise RuntimeError(f'{ckpt_dir} already exists at step {latest_step}. If you want to re-start '
                                           f'training, set `allow_restart=True`.')
                else:
                    params = model.init(jax_rng, batch_training)

                opt_state = optimizer.init(params)

            # Make sure parameters and opt_state are set.
            assert params is not None
            assert opt_state is not None

            params, train_metrics = training_step_fn(params, opt_state, batch_training)
            step += 1
            train_metrics_np = jax.device_get(train_metrics)

            # Log training metrics.
            if use_wandb:
                wandb.log(
                    {'Training {}'.format(k): v for (k, v) in train_metrics_np.items()},
                    step=step
                )

            # Start validation process.
            if step % eval_every_num_steps == 0:
                iterator_validation = jraph.dynamically_batch(
                    validation_data,
                    n_node=batch_max_num_nodes,
                    n_edge=batch_max_num_edges,
                    n_graph=batch_max_num_graphs
                )

                # Start iteration over validation batches.
                validation_metrics = []
                for graph_batch_validation in iterator_validation:
                    batch_validation = graph_to_batch_fn(graph_batch_validation)
                    batch_validation = jax.tree_map(jnp.array, batch_validation)

                    validation_metrics += [
                        validation_step_fn(
                            params,
                            batch_validation
                        )
                    ]

                validation_metrics_np = jax.device_get(validation_metrics)
                validation_metrics_np = {
                    k: np.mean([metrics[k] for metrics in validation_metrics]) for k in validation_metrics_np[0]
                }

                # Save checkpoint.
                ckpt_mngr.save(
                    step,
                    {'params': params},
                    metrics={'loss': validation_metrics_np['loss'].item()}
                )

                # Log to weights and bias.
                if use_wandb:
                    wandb.log({
                        f'Validation {k}': v for (k, v) in validation_metrics_np.items()},
                        step=step
                    )
            # Finished validation process.

    # Wait until checkpoint manager completes all save operations.
    ckpt_mngr.wait_until_finished()


def make_optimizer(
        name: str = 'adam',
        learning_rate: float = 1e-3,
        learning_rate_schedule: str = 'constant_schedule',
        learning_rate_schedule_args: Dict = dict(),
        num_of_nans_to_ignore: int = 0
):
    """Make optax optimizer.

    Args:
        name (str): Name of the optimizer. Defaults to the Adam optimizer.
        learning_rate (float): Learning rate.
        learning_rate_schedule (str): Learning rate schedule. Defaults to no schedule, meaning learning rate is
            held constant.
        learning_rate_schedule_args (dict): Arguments for the learning rate schedule.
        num_of_nans_to_ignore (int): Number of times NaNs are ignored during in the gradient step. Defaults to 0.

    Returns:

    """
    opt = getattr(optax, name)
    lr_schedule = getattr(optax, learning_rate_schedule)

    lr_schedule = lr_schedule(learning_rate, **learning_rate_schedule_args)
    opt = opt(lr_schedule)

    return optax.apply_if_finite(opt, max_consecutive_errors=num_of_nans_to_ignore)
