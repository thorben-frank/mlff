import jax.numpy as jnp
import jax

from typing import (Any, Callable, Dict, Tuple)
from flax.core.frozen_dict import FrozenDict

Array = Any
StackNetSparse = Any
LossFn = Callable[[FrozenDict, Dict[str, jnp.ndarray]], jnp.ndarray]
MetricFn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
DataTupleT = Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]
Derivative = Tuple[str, Tuple[str, str, Callable]]
ObservableFn = Callable[[FrozenDict, Dict[str, Array]], Dict[str, Array]]


def get_observable_fn_sparse(model: StackNetSparse, observable: str = None):
    """
    Get the observable function of a `model`. If no `observable_key` is specified, values for all implemented
    observables of the model are returned.

    Args:
        model (StackNet): A `StackNetSparse` module or a module that returns a dictionary of observables.
        observable (str): Observable name.

    Returns: Observable function,

    """
    if observable is None:
        def observable_fn(
                params,
                positions: jnp.ndarray,
                atomic_numbers: jnp.ndarray,
                idx_i: jnp.ndarray,
                idx_j: jnp.ndarray,
                total_charge: jnp.ndarray = None,
                num_unpaired_electrons: jnp.ndarray = None,
                cell: jnp.ndarray = None,
                cell_offset: jnp.ndarray = None,
                batch_segments: jnp.ndarray = None,
                node_mask: jnp.ndarray = None,
                graph_mask: jnp.ndarray = None,
                displacements: jnp.ndarray = None
        ):
            if batch_segments is None:
                assert graph_mask is None
                assert node_mask is None

                graph_mask = jnp.ones((1,)).astype(jnp.bool_)  # (1)
                node_mask = jnp.ones((len(atomic_numbers),)).astype(jnp.bool_)  # (num_nodes)
                batch_segments = jnp.zeros_like(atomic_numbers)  # (num_nodes)

            inputs = dict(
                positions=positions,
                displacements=displacements,
                atomic_numbers=atomic_numbers,
                idx_i=idx_i,
                idx_j=idx_j,
                total_charge=total_charge,
                num_unpaired_electrons=num_unpaired_electrons,
                cell=cell,
                cell_offset=cell_offset,
                batch_segments=batch_segments,
                node_mask=node_mask,
                graph_mask=graph_mask
            )
            return model.apply(params, inputs)
    else:
        def observable_fn(
                params,
                positions: jnp.ndarray,
                atomic_numbers: jnp.ndarray,
                idx_i: jnp.ndarray,
                idx_j: jnp.ndarray,
                total_charge: jnp.ndarray = None,
                num_unpaired_electrons: jnp.ndarray = None,
                cell: jnp.ndarray = None,
                cell_offset: jnp.ndarray = None,
                batch_segments: jnp.ndarray = None,
                node_mask: jnp.ndarray = None,
                graph_mask: jnp.ndarray = None,
                displacements: jnp.ndarray = None
        ):
            if batch_segments is None:
                assert graph_mask is None
                assert node_mask is None

                graph_mask = jnp.ones((1,)).astype(jnp.bool_)  # (1)
                node_mask = jnp.ones((len(positions),)).astype(jnp.bool_)  # (num_nodes)
                batch_segments = jnp.zeros_like(atomic_numbers)  # (num_nodes)

            inputs = dict(
                positions=positions,
                displacements=displacements,
                atomic_numbers=atomic_numbers,
                idx_i=idx_i,
                idx_j=idx_j,
                total_charge=total_charge,
                num_unpaired_electrons=num_unpaired_electrons,
                cell=cell,
                cell_offset=cell_offset,
                batch_segments=batch_segments,
                node_mask=node_mask,
                graph_mask=graph_mask
            )
            return dict(observable=model.apply(params, inputs)[observable])

    return observable_fn


def get_energy_and_force_fn_sparse(model: StackNetSparse):
    def energy_fn(params,
                  positions: jnp.ndarray,
                  atomic_numbers: jnp.ndarray,
                  idx_i: jnp.ndarray,
                  idx_j: jnp.ndarray,
                  total_charge: jnp.ndarray = None,
                  num_unpaired_electrons: jnp.ndarray = None,
                  cell: jnp.ndarray = None,
                  cell_offset: jnp.ndarray = None,
                  batch_segments: jnp.ndarray = None,
                  node_mask: jnp.ndarray = None,
                  graph_mask: jnp.ndarray = None,
                  total_charge: jnp.ndarray = None):
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
                      total_charge=total_charge,
                      num_unpaired_electrons=num_unpaired_electrons,
                      cell=cell,
                      cell_offset=cell_offset,
                      batch_segments=batch_segments,
                      node_mask=node_mask,
                      graph_mask=graph_mask,
                      total_charge=total_charge
                      )

        energy = model.apply(params, inputs)['energy']  # (num_graphs)
        energy = jnp.where(graph_mask, energy, jnp.asarray(0., dtype=energy.dtype))  # (num_graphs)
        return -jnp.sum(energy), energy  # (), (num_graphs)

    def energy_and_force_fn(params,
                            positions: jnp.ndarray,
                            atomic_numbers: jnp.ndarray,
                            idx_i: jnp.ndarray,
                            idx_j: jnp.ndarray,
                            total_charge: jnp.ndarray = None,
                            num_unpaired_electrons: jnp.ndarray = None,
                            cell: jnp.ndarray = None,
                            cell_offset: jnp.ndarray = None,
                            batch_segments: jnp.ndarray = None,
                            node_mask: jnp.ndarray = None,
                            graph_mask: jnp.ndarray = None,
                            *args,
                            **kwargs):
        (_, energy), forces = jax.value_and_grad(
            energy_fn,
            argnums=1,
            has_aux=True)(params,
                          positions,
                          atomic_numbers,
                          idx_i,
                          idx_j,
                          total_charge,
                          num_unpaired_electrons,
                          cell,
                          cell_offset,
                          batch_segments,
                          node_mask,
                          graph_mask
                          )
        
        return dict(energy=energy, forces=forces)
    
    # return energy_and_force_fn

    def energy_and_force_and_dipole_fn(params,
                            positions: jnp.ndarray,
                            atomic_numbers: jnp.ndarray,
                            idx_i: jnp.ndarray,
                            idx_j: jnp.ndarray,
                            cell: jnp.ndarray = None,
                            cell_offset: jnp.ndarray = None,
                            batch_segments: jnp.ndarray = None,
                            node_mask: jnp.ndarray = None,
                            graph_mask: jnp.ndarray = None,
                            *args,
                            **kwargs):
        (_, energy), forces = jax.value_and_grad(
            energy_fn,
            argnums=1,
            has_aux=True)(params,
                          positions,
                          atomic_numbers,
                          idx_i,
                          idx_j,
                          cell,
                          cell_offset,
                          batch_segments,
                          node_mask,
                          graph_mask
                          )

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
                cell=cell,
                cell_offset=cell_offset,
                batch_segments=batch_segments,
                node_mask=node_mask,
                graph_mask=graph_mask
                )

        dipole = model.apply(params, inputs)['dipole']  # (num_graphs)
        dipole = jnp.where(graph_mask, dipole, jnp.asarray(0., dtype=dipole.dtype))  # (num_graphs)
        return dict(energy=energy, forces=forces, dipole=dipole)

    def energy_and_force_and_dipole_and_hirsh_fn(params,
                            positions: jnp.ndarray,
                            atomic_numbers: jnp.ndarray,
                            idx_i: jnp.ndarray,
                            idx_j: jnp.ndarray,
                            cell: jnp.ndarray = None,
                            cell_offset: jnp.ndarray = None,
                            batch_segments: jnp.ndarray = None,
                            node_mask: jnp.ndarray = None,
                            graph_mask: jnp.ndarray = None,
                            total_charge: jnp.ndarray = None,
                            *args,
                            **kwargs):
        (_, energy), forces = jax.value_and_grad(
            energy_fn,
            argnums=1,
            has_aux=True)(params,
                          positions,
                          atomic_numbers,
                          idx_i,
                          idx_j,
                          cell,
                          cell_offset,
                          batch_segments,
                          node_mask,
                          graph_mask,
                          total_charge
                          )

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
                cell=cell,
                cell_offset=cell_offset,
                batch_segments=batch_segments,
                node_mask=node_mask,
                graph_mask=graph_mask,
                total_charge=total_charge
                )

        dipole = model.apply(params, inputs)['dipole']  # (num_graphs)
        dipole = jnp.where(graph_mask, dipole, jnp.asarray(0., dtype=dipole.dtype))  # (num_graphs)
        hirshfeld_ratios = model.apply(params, inputs)['hirshfeld_ratios']  # (num_graphs)
        hirshfeld_ratios = jnp.where(node_mask, hirshfeld_ratios, jnp.asarray(0., dtype=hirshfeld_ratios.dtype))  # (num_graphs)
        return dict(energy=energy, forces=forces, dipole=dipole, hirshfeld_ratios = hirshfeld_ratios)
        
    # return energy_and_force_and_dipole_fn
    return energy_and_force_and_dipole_and_hirsh_fn


# def get_energy_and_force_fn_sparse(model: StackNetSparse):
#     def energy_fn(params,
#                   positions: jnp.ndarray,
#                   atomic_numbers: jnp.ndarray,
#                   idx_i: jnp.ndarray,
#                   idx_j: jnp.ndarray,
#                   cell: jnp.ndarray = None,
#                   cell_offset: jnp.ndarray = None,
#                   batch_segments: jnp.ndarray = None,
#                   node_mask: jnp.ndarray = None,
#                   graph_mask: jnp.ndarray = None):
#         if batch_segments is None:
#             assert graph_mask is None
#             assert node_mask is None
#
#             graph_mask = jnp.ones((1,)).astype(jnp.bool_)  # (1)
#             node_mask = jnp.ones((len(positions),)).astype(jnp.bool_)  # (num_nodes)
#             batch_segments = jnp.zeros_like(atomic_numbers)  # (num_nodes)
#
#         inputs = dict(positions=positions,
#                       atomic_numbers=atomic_numbers,
#                       idx_i=idx_i,
#                       idx_j=idx_j,
#                       cell=cell,
#                       cell_offset=cell_offset,
#                       batch_segments=batch_segments,
#                       node_mask=node_mask,
#                       graph_mask=graph_mask
#                       )
#
#         energy = model.apply(params, inputs)['energy']  # (num_graphs)
#         energy = jnp.where(graph_mask, energy, jnp.asarray(0., dtype=energy.dtype))  # (num_graphs)
#         return -jnp.sum(energy), energy  # (), (num_graphs)
#
#     def energy_and_force_fn(params,
#                             positions: jnp.ndarray,
#                             atomic_numbers: jnp.ndarray,
#                             idx_i: jnp.ndarray,
#                             idx_j: jnp.ndarray,
#                             cell: jnp.ndarray = None,
#                             cell_offset: jnp.ndarray = None,
#                             batch_segments: jnp.ndarray = None,
#                             node_mask: jnp.ndarray = None,
#                             graph_mask: jnp.ndarray = None,
#                             *args,
#                             **kwargs):
#
#         # _, energy = energy_fn(
#         #     params,
#         #     positions,
#         #     atomic_numbers,
#         #     idx_i,
#         #     idx_j,
#         #     cell,
#         #     cell_offset,
#         #     batch_segments,
#         #     node_mask,
#         #     graph_mask
#         # )
#
#         forces, energy = jax.jacrev(energy_fn, argnums=1, has_aux=True)(
#             params,
#             positions,
#             atomic_numbers,
#             idx_i,
#             idx_j,
#             cell,
#             cell_offset,
#             batch_segments,
#             node_mask,
#             graph_mask
#         )
#
#         return dict(energy=energy, forces=forces)
#
#     return energy_and_force_fn