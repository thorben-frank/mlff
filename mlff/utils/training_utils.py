import clu.metrics as clu_metrics
from flax import traverse_util
import jraph
import jax
import jax.numpy as jnp
import numpy as np
import optax
from orbax import checkpoint
from pathlib import Path
from typing import Any, Callable, Dict, Sequence
import wandb

property_to_mask = {
    'energy': 'graph_mask',
    'stress': 'graph_mask',
    'forces': 'node_mask',
    'dipole': 'graph_mask',
    'dipole_vec': 'graph_mask_expanded',
    'hirshfeld_ratios': 'node_mask',
    'dispersion_energy': 'graph_mask',
    'electrostatic_energy': 'graph_mask'
}

def print_metrics(epoch, eval_metrics):
    formatted_output = f"{epoch}: "
    for key in list(eval_metrics.keys()):
        formatted_output += f"{key}={eval_metrics[key]:.4f}, "
    formatted_output = formatted_output.rstrip(", ")
    return formatted_output

def scaled_mse_loss(y, y_label, scale, mask):
    full_mask = ~jnp.isnan(y_label) & jnp.expand_dims(mask, [y_label.ndim - 1 - o for o in range(0, y_label.ndim - 1)])
    denominator = full_mask.sum().astype(y.dtype)
    mse = (
            jnp.sum(
                2 * scale * optax.l2_loss(
                    jnp.where(full_mask, y, 0).reshape(-1),
                    jnp.where(full_mask, y_label, 0).reshape(-1),
                )
            )
            / denominator
    )
    return mse


def graph_mse_loss(y, y_label, batch_segments, graph_mask, scale):
    del batch_segments

    assert y.shape == y_label.shape

    full_mask = ~jnp.isnan(
        y_label
    ) & jnp.expand_dims(
        graph_mask, [y_label.ndim - 1 - o for o in range(0, y_label.ndim - 1)]
    )
    denominator = full_mask.sum().astype(y.dtype)
    mse = (
            jnp.sum(
                2 * scale * optax.l2_loss(
                    jnp.where(full_mask, y, 0).reshape(-1),
                    jnp.where(full_mask, y_label, 0).reshape(-1),
                )
            )
            / denominator
    )
    return mse


def node_mse_loss(y, y_label, batch_segments, graph_mask, scale):

    assert y.shape == y_label.shape

    num_graphs = graph_mask.sum().astype(y.dtype)  # ()

    squared = 2 * optax.l2_loss(
        predictions=y,
        targets=y_label,
    )  # same shape as y

    # sum up the l2_losses for node properties along the non-leading dimension. For e.g. scalar node quantities
    # this does not have any effect, but e.g. for vectorial and tensorial node properties one averages over all
    # additional non-leading dimension. E.g. for forces this corresponds to taking mean over x, y, z component.
    node_mean_squared = squared.reshape(len(squared), -1).mean(axis=-1)  # (num_nodes)

    per_graph_mse = jraph.segment_mean(
        data=node_mean_squared,
        segment_ids=batch_segments,
        num_segments=len(graph_mask)
    )  # (num_graphs)

    # Set contributions from padding graphs to zero.
    per_graph_mse = jnp.where(
        graph_mask,
        per_graph_mse,
        jnp.asarray(0., dtype=per_graph_mse.dtype)
    )  # (num_graphs)

    # Calculate mean and scale.
    mse = scale * jnp.sum(per_graph_mse) / num_graphs  # ()

    return mse


property_to_loss = {
    'energy': graph_mse_loss,
    'stress': graph_mse_loss,
    'forces': node_mse_loss,
}


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
            # _l = scaled_mse_loss(
            #     y=outputs_predict[target],
            #     y_label=outputs_true[target],
            #     scale=_scales[target],
            #     mask=inputs[property_to_mask[target]]
            # )
            target_loss_fn = property_to_loss[target]
            _l = target_loss_fn(
                y=outputs_predict[target],
                y_label=outputs_true[target],
                scale=_scales[target],
                batch_segments=inputs['batch_segments'],
                graph_mask=inputs['graph_mask'],
            )

            loss += weights[target] * _l
            metrics.update({f'{target}_mse': _l / _scales[target].mean()})

        loss = jnp.reshape(loss, ())
        metrics.update({'loss': loss})

        return loss, metrics

    return loss_fn


def make_training_step_fn(optimizer, loss_fn, log_gradient_values):
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
        # if log_gradient_values:
        #     metrics['grad_norm'] = unfreeze(jax.tree_map(lambda x: jnp.linalg.norm(x.reshape(-1), axis=0), grads))
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params=params, updates=updates)
        metrics['grad_norm'] = optax.global_norm(grads)
        return params, opt_state, metrics

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
        params=None,
        num_epochs: int = 100,
        ckpt_dir: str = None,
        ckpt_manager_options: dict = None,
        eval_every_num_steps: int = 1000,
        allow_restart: bool = False,
        training_seed: int = 0,
        model_seed: int = 0,
        use_wandb: bool = True,
        log_gradient_values: bool = False
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
        params: Parameters to start from during training. If not given, either new parameters are initialized randomly
            or loaded from ckpt_dir if the checkpoint already exists and `allow_restart=True`.
        num_epochs (int): Number of training epochs.
        ckpt_dir (str): Checkpoint path.
        ckpt_manager_options (dict): Checkpoint manager options.
        eval_every_num_steps (int): Evaluate the metrics every num-th step
        allow_restart: Restarts from existing checkpoints are allowed.
        training_seed (int): Random seed for shuffling of training data.
        model_seed (int): Random seed for model initialization.
        use_wandb (bool): Log statistics to WeightsAndBias. If true, wandb.init() must be called before call to fit().
        log_gradient_values (bool): Gradient values for each set of weights is logged.
    Returns:

    """
    numpy_rng = np.random.RandomState(seed=training_seed)
    jax_rng = jax.random.PRNGKey(seed=model_seed)

    # Create checkpoint directory.
    ckpt_dir = Path(ckpt_dir).expanduser().resolve()
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
        item_names=('params', ),
        options=options
    )

    training_step_fn = make_training_step_fn(optimizer, loss_fn, log_gradient_values)
    validation_step_fn = make_validation_step_fn(loss_fn)

    processed_graphs = 0
    processed_nodes = 0
    step = 0

    opt_state = None
    for epoch in range(num_epochs):
        # Shuffle the training data.
        numpy_rng.shuffle(training_data)

        # Create batched graphs from list of graphs.
        iterator_training = jraph.dynamically_batch(
            training_data,
            n_node=batch_max_num_nodes,
            n_edge=batch_max_num_edges,
            n_graph=batch_max_num_graphs,
            n_pairs=batch_max_num_nodes*(batch_max_num_nodes-1)//(batch_max_num_graphs//2),
        )

        # Start iteration over batched graphs.
        for graph_batch_training in iterator_training:
            batch_training = graph_to_batch_fn(graph_batch_training)
            processed_graphs += batch_training['num_of_non_padded_graphs']
            processed_nodes += batch_max_num_nodes - jraph.get_number_of_padding_with_graphs_nodes(graph_batch_training)
            # Training data is numpy arrays so we now transform them to jax.numpy arrays.
            batch_training = jax.tree_map(jnp.array, batch_training)

            # If params are None (in the first step), initialize the parameters or load from existing checkpoint.
            if params is None:
                # Check if checkpoint already exists.
                latest_step = ckpt_mngr.latest_step()
                if latest_step is not None:
                    if allow_restart:
                        params = ckpt_mngr.restore(
                            latest_step,
                            args=checkpoint.args.Composite(params=checkpoint.args.StandardRestore())
                        )['params']
                        step += latest_step
                        print(f'Re-start training from {latest_step}.')
                    else:
                        raise RuntimeError(f'{ckpt_dir} already exists at step {latest_step}. If you want to re-start '
                                           f'training, set `allow_restart=True`.')
                else:
                    params = model.init(jax_rng, batch_training)

            # If optimizer state is None (in the first step), initialize from the parameter pyTree.
            if opt_state is None:
                opt_state = optimizer.init(params)

            # Make sure parameters and opt_state are set.
            assert params is not None
            assert opt_state is not None

            params, opt_state, train_metrics = training_step_fn(params, opt_state, batch_training)
            step += 1
            train_metrics_np = jax.device_get(train_metrics)

            # Log training metrics.
            if use_wandb:
                wandb.log(
                    {f'train_{k}': v for (k, v) in train_metrics_np.items()},
                    step=step
                )

            # Print train metrics
            # print(print_metrics(f"train_{epoch}_{step}:", train_metrics_np))

            # Start validation process.
            if step % eval_every_num_steps == 0:
                iterator_validation = jraph.dynamically_batch(
                    validation_data,
                    n_node=batch_max_num_nodes,
                    n_edge=batch_max_num_edges,
                    n_graph=batch_max_num_graphs,
                    n_pairs=batch_max_num_nodes*(batch_max_num_nodes-1)//(batch_max_num_graphs//2),
                )

                # Start iteration over validation batches.
                eval_metrics: Any = None
                eval_collection: Any = None
                for graph_batch_validation in iterator_validation:
                    batch_validation = graph_to_batch_fn(graph_batch_validation)
                    batch_validation = jax.tree_map(jnp.array, batch_validation)

                    eval_out = validation_step_fn(
                        params,
                        batch_validation
                    )
                    # The metrics are created dynamically during the first evaluation batch, since we aim to support
                    # all kinds of targets beyond energies and forces at some point.
                    if eval_collection is None:
                        eval_collection = clu_metrics.Collection.create(
                            **{k: clu_metrics.Average.from_output(f'{k}') for k in eval_out.keys()})

                    eval_metrics = (
                        eval_collection.single_from_model_output(**eval_out)
                        if eval_metrics is None
                        else eval_metrics.merge(eval_collection.single_from_model_output(**eval_out))
                    )

                eval_metrics = eval_metrics.compute()

                # Convert to dict to log with weights and bias.
                eval_metrics = {
                    f'eval_{k}': float(v) for k, v in eval_metrics.items()
                }

                # Print eval_metrics
                print(print_metrics(f"val_{epoch}_{step}:", eval_metrics))

                # Save checkpoint.
                ckpt_mngr.save(
                    step,
                    args=checkpoint.args.Composite(params=checkpoint.args.StandardSave(params)),
                    metrics={'loss': eval_metrics['eval_loss']}
                )

                # Log to weights and bias.
                if use_wandb:
                    wandb.log(
                        eval_metrics,
                        step=step
                    )
            # Finished validation process.

    # Wait until checkpoint manager completes all save operations.
    ckpt_mngr.wait_until_finished()


def fit_from_iterator(
        model,
        optimizer,
        loss_fn,
        graph_to_batch_fn,
        training_iterator,
        validation_iterator,
        batch_max_num_nodes,
        batch_max_num_edges,
        batch_max_num_graphs,
        params=None,
        ckpt_dir: str = None,
        ckpt_manager_options: dict = None,
        eval_every_num_steps: int = 1000,
        allow_restart: bool = False,
        training_seed: int = 0,
        model_seed: int = 0,
        use_wandb: bool = True,
        log_gradient_values: bool = False
):
    """
    Fit model.

    Args:
        model: flax module.
        optimizer: optax optimizer.
        loss_fn (Callable): The loss function. Gradient is computed wrt to this function.
        graph_to_batch_fn (Callable): Function that takes a batched graph and returns a batch for the loss_fn.
        training_iterator (): Iterator yielding jraph.GraphTuples.
        validation_iterator (): Iterator yielding jraph.GraphTuples.
        batch_max_num_nodes (int): Maximal number of nodes per batch.
        batch_max_num_edges (int): Maximal number of edges per batch.
        batch_max_num_graphs (int): Maximal number of graphs per batch.
        params: Parameters to start from during training. If not given, either new parameters are initialized randomly
            or loaded from ckpt_dir if the checkpoint already exists and `allow_restart=True`.
        ckpt_dir (str): Checkpoint path.
        ckpt_manager_options (dict): Checkpoint manager options.
        eval_every_num_steps (int): Evaluate the metrics every num-th step
        allow_restart: Restarts from existing checkpoints are allowed.
        training_seed (int): Random seed for shuffling of training data.
        model_seed (int): Random seed for model initialization.
        use_wandb (bool): Log statistics to WeightsAndBias. If true, wandb.init() must be called before call to fit().
        log_gradient_values (bool): Gradient values for each set of weights is logged.
    Returns:

    """
    # numpy_rng = np.random.RandomState(seed=training_seed)
    jax_rng = jax.random.PRNGKey(seed=model_seed)

    # Create checkpoint directory.
    ckpt_dir = Path(ckpt_dir).expanduser().resolve()
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
        item_names=('params', ),
        options=options
    )

    training_step_fn = make_training_step_fn(optimizer, loss_fn, log_gradient_values)
    validation_step_fn = make_validation_step_fn(loss_fn)

    processed_graphs = 0
    processed_nodes = 0
    step = 0

    opt_state = None

    # Create batched graphs from iterator over single graphs.
    training_iterator_batched = jraph.dynamically_batch(
        training_iterator.as_numpy_iterator(),
        n_node=batch_max_num_nodes,
        n_edge=batch_max_num_edges,
        n_graph=batch_max_num_graphs
    )

    # Start iteration over batched graphs.
    for graph_batch_training in training_iterator_batched:
        batch_training = graph_to_batch_fn(graph_batch_training)
        processed_graphs += batch_training['num_of_non_padded_graphs']
        processed_nodes += batch_max_num_nodes - jraph.get_number_of_padding_with_graphs_nodes(graph_batch_training)
        # Training data is numpy arrays so we now transform them to jax.numpy arrays.
        batch_training = jax.tree_map(jnp.array, batch_training)

        # If params are None (in the first step), initialize the parameters or load from existing checkpoint.
        if params is None:
            # Check if checkpoint already exists.
            latest_step = ckpt_mngr.latest_step()
            if latest_step is not None:
                if allow_restart:
                    params = ckpt_mngr.restore(
                        latest_step,
                        args=checkpoint.args.Composite(params=checkpoint.args.StandardRestore())
                    )['params']
                    step += latest_step
                    print(f'Re-start training from {latest_step}.')
                else:
                    raise RuntimeError(f'{ckpt_dir} already exists at step {latest_step}. If you want to re-start '
                                       f'training, set `allow_restart=True`.')
            else:
                params = model.init(jax_rng, batch_training)

        # If optimizer state is None (in the first step), initialize from the parameter pyTree.
        if opt_state is None:
            opt_state = optimizer.init(params)

        # Make sure parameters and opt_state are set.
        assert params is not None
        assert opt_state is not None

        params, opt_state, train_metrics = training_step_fn(params, opt_state, batch_training)
        step += 1
        train_metrics_np = jax.device_get(train_metrics)

        # Log training metrics.
        if use_wandb:
            wandb.log(
                {f'train_{k}': v for (k, v) in train_metrics_np.items()},
                step=step
            )

        # Start validation process.
        if step % eval_every_num_steps == 0:
            validation_iterator_batched = jraph.dynamically_batch(
                validation_iterator.as_numpy_iterator(),
                n_node=batch_max_num_nodes,
                n_edge=batch_max_num_edges,
                n_graph=batch_max_num_graphs
            )

            # Start iteration over validation batches.
            eval_metrics: Any = None
            eval_collection: Any = None
            for graph_batch_validation in validation_iterator_batched:
                batch_validation = graph_to_batch_fn(graph_batch_validation)
                batch_validation = jax.tree_map(jnp.array, batch_validation)

                eval_out = validation_step_fn(
                    params,
                    batch_validation
                )
                # The metrics are created dynamically during the first evaluation batch, since we aim to support
                # all kinds of targets beyond energies and forces at some point.
                if eval_collection is None:
                    eval_collection = clu_metrics.Collection.create(
                        **{k: clu_metrics.Average.from_output(f'{k}') for k in eval_out.keys()})

                eval_metrics = (
                    eval_collection.single_from_model_output(**eval_out)
                    if eval_metrics is None
                    else eval_metrics.merge(eval_collection.single_from_model_output(**eval_out))
                )

            eval_metrics = eval_metrics.compute()

            # Convert to dict to log with weights and bias.
            eval_metrics = {
                f'eval_{k}': float(v) for k, v in eval_metrics.items()
            }

            # Save checkpoint.
            ckpt_mngr.save(
                step,
                args=checkpoint.args.Composite(params=checkpoint.args.StandardSave(params)),
                metrics={'loss': eval_metrics['eval_loss']}
            )

            # Log to weights and bias.
            if use_wandb:
                wandb.log(
                    eval_metrics,
                    step=step
                )
        # Finished validation process.

    # Wait until checkpoint manager completes all save operations.
    ckpt_mngr.wait_until_finished()


def make_optimizer(
        name: str = 'adam',
        optimizer_args: Dict = dict(),
        learning_rate: float = 1e-3,
        learning_rate_schedule: str = 'constant_schedule',
        learning_rate_schedule_args: Dict = dict(),
        gradient_clipping: str = 'identity',
        gradient_clipping_args: Dict = dict(),
        num_of_nans_to_ignore: int = 0
):
    """Make optax optimizer.

    Args:
        name (str): Name of the optimizer. Defaults to the Adam optimizer.
        optimizer_args (dict): Arguments passed to the optimizer.
        learning_rate (float): Learning rate.
        learning_rate_schedule (str): Learning rate schedule. Defaults to no schedule, meaning learning rate is
            held constant.
        learning_rate_schedule_args (dict): Arguments for the learning rate schedule.
        num_of_nans_to_ignore (int): Number of times NaNs are ignored during in the gradient step. Defaults to 0.
        gradient_clipping (str): Gradient clipping to apply.
        gradient_clipping_args (dict): Arguments to the gradient clipping to apply.
    Returns:

    """
    opt = getattr(optax, name)
    lr_schedule = getattr(optax, learning_rate_schedule)

    lr_schedule = lr_schedule(learning_rate, **learning_rate_schedule_args)
    opt = opt(lr_schedule, **optimizer_args)

    clip_transform = getattr(optax, gradient_clipping)
    clip_transform = clip_transform(**gradient_clipping_args)

    return optax.chain(
        clip_transform,
        optax.zero_nans(),
        opt
    )

    # return optax.apply_if_finite(opt, max_consecutive_errors=num_of_nans_to_ignore)
    # return opt