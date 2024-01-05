import jax
import jax.numpy as jnp
import jraph
from mlff.utils import training_utils
import numpy as np
from orbax import checkpoint
from pathlib import Path
from typing import Dict
import wandb


def run_training_sparse(
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
        eval_every_t: int = 1,
        allow_restart=False,
        training_seed: int = 0,
        model_seed: int = 0,
        use_wandb: bool = True,
        wandb_init_args: Dict = None
):
    """
    Run training.

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
        eval_every_t (int): Evaluate the metrics every t-th step
        allow_restart: Restarts from existing checkpoints are allowed.
        training_seed (int): Random seed for shuffling of training data.
        model_seed (int): Random seed for model initialization.
        use_wandb (bool): Log statistics to WeightsAndBias.
        wandb_init_args (Dict): Arguments passed to `wandb.init()`.

    Returns:

    """
    numpy_rng = np.random.RandomState(seed=training_seed)
    jax_rng = jax.random.PRNGKey(seed=model_seed)

    # Initialize weights and bias run.
    if use_wandb:
        wandb.init(**wandb_init_args) if wandb_init_args is not None else wandb.init()

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

    training_step_fn = training_utils.make_training_step_fn(optimizer, loss_fn)
    validation_step_fn = training_utils.make_validation_step_fn(loss_fn)

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
            if step % eval_every_t == 0:
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
