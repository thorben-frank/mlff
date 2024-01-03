import jax.numpy as jnp
import jax
import logging

from typing import (Any, Callable, Dict, Sequence, Tuple)
from flax.core.frozen_dict import FrozenDict

# logging.basicConfig(level=logging.INFO)

Array = Any
StackNetSparse = Any
LossFn = Callable[[FrozenDict, Dict[str, jnp.ndarray]], jnp.ndarray]
MetricFn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
DataTupleT = Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]
Derivative = Tuple[str, Tuple[str, str, Callable]]
ObservableFn = Callable[[FrozenDict, Dict[str, Array]], Dict[str, Array]]


def get_observable_fn_sparse(model: StackNetSparse, observable_key: str = None) -> ObservableFn:
    """
    Get the observable function of a `model`. If no `observable_key` is specified, values for all implemented
    observables of the model are returned.

    Args:
        model (StackNet): A `StackNetSparse` module or a module that returns a dictionary of observables.
        observable_key (str): Observable key.

    Returns: Observable function,

    """
    if observable_key is None:
        def observable_fn(p, x):
            return model.apply(p, x)
    else:
        def observable_fn(p, x):
            return {observable_key: model.apply(p, x)[observable_key]}

    return observable_fn


def get_energy_and_force_fn_sparse(model: StackNetSparse):
    def energy_fn(params,
                  positions: jnp.ndarray,
                  atomic_numbers: jnp.ndarray,
                  idx_i: jnp.ndarray,
                  idx_j: jnp.ndarray,
                  batch_segments: jnp.ndarray = None,
                  node_mask: jnp.ndarray = None,
                  graph_mask: jnp.ndarray = None):
        if batch_segments is None:
            assert graph_mask is None
            assert node_mask is None

            graph_mask = jnp.ones((1,)).astype(jnp.bool_)  # (1)
            node_mask = jnp.ones((len(positions),)).astype(jnp.bool_)  # (num_nodes)
            batch_segments = jnp.zeros_like(atomic_numbers)  # (num_nodes)

        inputs = dict(positions=positions,
                      atomic_numbers=atomic_numbers,
                      idx_i=idx_i,
                      idx_j=idx_j,
                      batch_segments=batch_segments,
                      node_mask=node_mask,
                      graph_mask=graph_mask
                      )

        energy = model.apply(params, inputs)['energy']  # (num_graphs)
        energy = jnp.where(graph_mask, energy, jnp.asarray(0., dtype=energy.dtype))  # (num_graphs)
        return -jnp.sum(energy), energy  # (), (num_graphs)

    def energy_and_force_fn(params,
                            positions: jnp.ndarray,
                            atomic_numbers: jnp.ndarray,
                            idx_i: jnp.ndarray,
                            idx_j: jnp.ndarray,
                            batch_segments: jnp.ndarray = None,
                            node_mask: jnp.ndarray = None,
                            graph_mask: jnp.ndarray = None):
        (_, energy), forces = jax.value_and_grad(
            energy_fn,
            argnums=1,
            has_aux=True)(params,
                          positions,
                          atomic_numbers,
                          idx_i,
                          idx_j,
                          batch_segments,
                          node_mask,
                          graph_mask
                          )

        return dict(energy=energy, forces=forces)

    return energy_and_force_fn
