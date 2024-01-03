import pytest

import jax.numpy as jnp
import numpy.testing as npt

import jraph

from mlff.utils import jraph_utils

graph1 = jraph.GraphsTuple(
    nodes=dict(
        positions=jnp.array(
            [
                [0., 0., 0.],
                [0., -2., 0.],
                [1., 0.5, 0.]
            ]
        ),
        forces=jnp.ones((3, 3)),
    ),
    receivers=jnp.array([0, 0, 1, 2]),
    senders=jnp.array([1, 2, 0, 0]),
    globals=dict(energy=jnp.array([1])),
    edges=None,
    n_node=jnp.array([3]),
    n_edge=jnp.array([4])
)

graph2 = jraph.GraphsTuple(
    nodes=dict(
        positions=jnp.array(
            [
                [1., 1., 1.],
                [2., 2., 2.],
            ]
        ),
        forces=jnp.ones((2, 3))*2,
    ),
    receivers=jnp.array([0, 1]),
    senders=jnp.array([1, 0]),
    globals=dict(energy=jnp.array([2])),
    edges=None,
    n_node=jnp.array([2]),
    n_edge=jnp.array([2])
)

graph3 = jraph.GraphsTuple(
    nodes=dict(
        positions=jnp.array(
            [
                [0., 1., 2.],
                [1., 1., 1.],
                [2., 2., 2.],
                [0., 0., 0.],
            ]
        ),
        forces=jnp.ones((4, 3))*3,
    ),
    receivers=jnp.array([0, 0, 1, 1, 2, 3]),
    senders=jnp.array([1, 3, 0, 2, 1, 0]),
    globals=dict(energy=jnp.array([3])),
    edges=None,
    n_node=jnp.array([4]),
    n_edge=jnp.array([6])
)


@pytest.mark.parametrize("max_num_graphs", [2, 3, 4])
def test_batch_info_fn(max_num_graphs):
    max_num_nodes = 11
    max_num_edges = 15
    batched_graphs = jraph.dynamically_batch(
        [graph1, graph2, graph3],
        n_node=max_num_nodes,
        n_edge=max_num_edges,
        n_graph=max_num_graphs
    )

    if max_num_graphs == 2:
        g = next(batched_graphs)
        batch_info = jraph_utils.batch_info_fn(g)
        npt.assert_allclose(
            batch_info.get('batch_segments'),
            jnp.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])
        )
        npt.assert_allclose(batch_info.get('node_mask'), jraph.get_node_padding_mask(g))
        npt.assert_allclose(batch_info.get('graph_mask'), jraph.get_graph_padding_mask(g))
        npt.assert_equal(batch_info.get('num_of_non_padded_graphs').item(), 1)

        g = next(batched_graphs)
        batch_info = jraph_utils.batch_info_fn(g)
        npt.assert_allclose(
            batch_info.get('batch_segments'),
            jnp.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        )
        npt.assert_allclose(batch_info.get('node_mask'), jraph.get_node_padding_mask(g))
        npt.assert_allclose(batch_info.get('graph_mask'), jraph.get_graph_padding_mask(g))
        npt.assert_equal(batch_info.get('num_of_non_padded_graphs').item(), 1)

        g = next(batched_graphs)
        batch_info = jraph_utils.batch_info_fn(g)
        npt.assert_allclose(
            batch_info.get('batch_segments'),
            jnp.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
        )
        npt.assert_allclose(batch_info.get('node_mask'), jraph.get_node_padding_mask(g))
        npt.assert_allclose(batch_info.get('graph_mask'), jraph.get_graph_padding_mask(g))
        npt.assert_equal(batch_info.get('num_of_non_padded_graphs').item(), 1)

        with npt.assert_raises(StopIteration):
            next(batched_graphs)

    if max_num_graphs == 3:
        g = next(batched_graphs)
        batch_info = jraph_utils.batch_info_fn(g)
        npt.assert_allclose(
            batch_info.get('batch_segments'),
            jnp.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2])
        )
        npt.assert_allclose(batch_info.get('node_mask'), jraph.get_node_padding_mask(g))
        npt.assert_allclose(batch_info.get('graph_mask'), jraph.get_graph_padding_mask(g))
        npt.assert_equal(batch_info.get('num_of_non_padded_graphs').item(), 2)

        g = next(batched_graphs)
        batch_info = jraph_utils.batch_info_fn(g)
        npt.assert_allclose(
            batch_info.get('batch_segments'),
            jnp.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])
        )
        npt.assert_allclose(batch_info.get('node_mask'), jraph.get_node_padding_mask(g))
        npt.assert_allclose(batch_info.get('graph_mask'), jraph.get_graph_padding_mask(g))
        npt.assert_equal(batch_info.get('num_of_non_padded_graphs').item(), 1)

        with npt.assert_raises(StopIteration):
            next(batched_graphs)

    if max_num_graphs == 4:
        g = next(batched_graphs)
        batch_info = jraph_utils.batch_info_fn(g)
        npt.assert_allclose(
            batch_info.get('batch_segments'),
            jnp.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3])
        )
        npt.assert_allclose(batch_info.get('node_mask'), jraph.get_node_padding_mask(g))
        npt.assert_allclose(batch_info.get('graph_mask'), jraph.get_graph_padding_mask(g))
        npt.assert_equal(batch_info.get('num_of_non_padded_graphs').item(), 3)

        with npt.assert_raises(StopIteration):
            next(batched_graphs)
