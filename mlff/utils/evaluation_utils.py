import jax
import jax.numpy as jnp
import jraph
import numpy as np
from mlff.nn.stacknet.observable_function_sparse import get_energy_and_force_fn_sparse


def evaluate(
        model,
        params,
        graph_to_batch_fn,
        testing_data,
        testing_targets,
        batch_max_num_nodes,
        batch_max_num_edges,
        batch_max_num_graphs
):
    """Evaluate a model given its params on the testing data.

    Args:
        model (): The FLAX model.
        params (): The model parameters as PyTree.
        graph_to_batch_fn (): Function that takes a `jraph.GraphsTuple` and returns the input to the
            observable function.
        testing_data (): The testing data as list of `jraph.GraphsTuple`.
        testing_targets (): The targets for which the metrics should be calculated.
        batch_max_num_nodes (): Maximal number of nodes per batch.
        batch_max_num_edges (): Maximal number of edges per batch.
        batch_max_num_graphs (): Maximal number of graphs oer batch.

    Returns:
        The metrics on testing data.
    """

    obs_fn = jax.jit(
        get_energy_and_force_fn_sparse(model)
    )

    iterator_testing = jraph.dynamically_batch(
        testing_data,
        n_node=batch_max_num_nodes,
        n_edge=batch_max_num_edges,
        n_graph=batch_max_num_graphs
    )

    # Start iteration over validation batches.
    testing_metrics = []
    for graph_batch_testing in iterator_testing:
        batch_testing = graph_to_batch_fn(graph_batch_testing)
        batch_testing = jax.tree_map(jnp.array, batch_testing)

        node_mask = batch_testing['node_mask']
        graph_mask = batch_testing['graph_mask']

        inputs = {k: v for (k, v) in batch_testing.items() if k not in testing_targets}
        output_prediction = obs_fn(params, **inputs)

        metrics_dict = {}
        for t in testing_targets:
            if t == 'energy':
                msk = graph_mask
            elif t == 'forces':
                msk = node_mask
            elif t == 'stress':
                msk = graph_mask
            else:
                raise ValueError(f"Evaluate not implemented for target={t}.")

            metrics_dict[f"{t}_mae"] = calculate_mae(
                y_predicted=output_prediction[t], y_true=batch_testing[t], msk=msk
            ),
            metrics_dict[f"{t}_mse"] = calculate_mse(
                y_predicted=output_prediction[t], y_true=batch_testing[t], msk=msk
            ),

            testing_metrics += [metrics_dict]

    testing_metrics_np = jax.device_get(testing_metrics)
    testing_metrics_np = {
        k: np.mean([m[k] for m in testing_metrics_np]) for k in testing_metrics_np[0]
    }
    for t in testing_targets:
        testing_metrics_np[f'{t}_rmse'] = np.sqrt(testing_metrics_np[f'{t}_mse'])
    return testing_metrics_np


def calculate_mse(y_predicted, y_true, msk):
    assert y_predicted.shape == y_true.shape
    assert len(y_predicted) == len(y_true) == len(msk)

    return jnp.square(
        y_predicted[msk] - y_true[msk]
    ).mean()


def calculate_mae(y_predicted, y_true, msk):
    assert y_predicted.shape == y_true.shape
    assert len(y_predicted) == len(y_true) == len(msk)

    return jnp.abs(
        y_predicted[msk] - y_true[msk]
    ).mean()
