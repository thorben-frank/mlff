import jax.numpy as jnp
import jax

from typing import (Any, Callable, Dict, Tuple)
from flax.core.frozen_dict import FrozenDict
from mlff.masking.mask import safe_scale

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
                displacements: jnp.ndarray = None,
                graph_mask_expanded: jnp.ndarray = None,
                total_charge: jnp.ndarray = None,
                hirsh_bool: jnp.ndarray = None,
                i_pairs: jnp.ndarray = None,
                j_pairs: jnp.ndarray = None,
                # d_ij_all: jnp.ndarray = None,
                pair_mask: jnp.ndarray = None,
                batch_segments_pairs: jnp.ndarray = None,
        ):
            if batch_segments is None:
                assert graph_mask is None
                assert node_mask is None
                assert graph_mask_expanded is None

                graph_mask = jnp.ones((1,)).astype(jnp.bool_)  # (1)
                node_mask = jnp.ones((len(atomic_numbers),)).astype(jnp.bool_)  # (num_nodes)
                batch_segments = jnp.zeros_like(atomic_numbers)  # (num_nodes)
                graph_mask_expanded = jnp.ones((1,3)).astype(jnp.bool_)  # (1,3)

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
                graph_mask=graph_mask,
                graph_mask_expanded=graph_mask_expanded,
                total_charge=total_charge,
                hirsh_bool=hirsh_bool,
                i_pairs=i_pairs,
                j_pairs=j_pairs,
                # d_ij_all=d_ij_all,
                pair_mask=pair_mask,
                batch_segments_pairs=batch_segments_pairs,
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
                displacements: jnp.ndarray = None,
                graph_mask_expanded: jnp.ndarray = None,
                total_charge: jnp.ndarray = None,
                hirsh_bool: jnp.ndarray = None,
                i_pairs: jnp.ndarray = None,
                j_pairs: jnp.ndarray = None,
                # d_ij_all: jnp.ndarray = None,
                pair_mask: jnp.ndarray = None,
                batch_segments_pairs: jnp.ndarray = None,
        ):
            if batch_segments is None:
                assert graph_mask is None
                assert graph_mask_expanded is None
                assert node_mask is None

                graph_mask = jnp.ones((1,)).astype(jnp.bool_)  # (1)
                graph_mask_expanded = jnp.ones((1,3)).astype(jnp.bool_)  # (1,3)
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
                graph_mask=graph_mask,
                graph_mask_expanded=graph_mask_expanded,
                total_charge=total_charge,
                hirsh_bool=hirsh_bool,
                i_pairs=i_pairs,
                j_pairs=j_pairs,
                # d_ij_all=d_ij_all,
                pair_mask=pair_mask,
                batch_segments_pairs=batch_segments_pairs,
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
                  graph_mask_expanded: jnp.ndarray = None,
                  total_charge: jnp.ndarray = None,
                  hirsh_bool: jnp.ndarray = None,
                  i_pairs: jnp.ndarray = None,
                  j_pairs: jnp.ndarray = None,
                #   d_ij_all: jnp.ndarray = None,
                  pair_mask: jnp.ndarray = None,
                  batch_segments_pairs: jnp.ndarray = None,
                  dummmy: jnp.ndarray = None):
        if batch_segments is None:
            assert graph_mask is None
            assert graph_mask_expanded is None
            assert node_mask is None

            graph_mask = jnp.ones((1,)).astype(jnp.bool_)  # (1)
            graph_mask_expanded = jnp.ones((1,3)).astype(jnp.bool_)  # (1,3)
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
                      graph_mask_expanded=graph_mask_expanded,
                      total_charge=total_charge,
                      hirsh_bool=hirsh_bool,
                      i_pairs=i_pairs,
                      j_pairs=j_pairs,
                    #   d_ij_all=d_ij_all,
                      pair_mask=pair_mask,
                      batch_segments_pairs=batch_segments_pairs,
                      dummmy = dummmy
                      )

        energy = model.apply(params, inputs)['energy']  # (num_graphs)
        energy = safe_scale(energy, graph_mask)
        return -jnp.sum(energy), energy  # (), (num_graphs)

    # def energy_and_force_fn(params,
    #                         positions: jnp.ndarray,
    #                         atomic_numbers: jnp.ndarray,
    #                         idx_i: jnp.ndarray,
    #                         idx_j: jnp.ndarray,
                            # total_charge: jnp.ndarray = None,
                            # num_unpaired_electrons: jnp.ndarray = None,
    #                         cell: jnp.ndarray = None,
    #                         cell_offset: jnp.ndarray = None,
    #                         batch_segments: jnp.ndarray = None,
    #                         node_mask: jnp.ndarray = None,
    #                         graph_mask: jnp.ndarray = None,
    #                         *args,
    #                         **kwargs):
    #     (_, energy), forces = jax.value_and_grad(
    #         energy_fn,
    #         argnums=1,
    #         has_aux=True)(params,
    #                       positions,
    #                       atomic_numbers,
    #                       idx_i,
    #                       idx_j,
                          total_charge,
                          num_unpaired_electrons,
    #                       cell,
    #                       cell_offset,
    #                       batch_segments,
    #                       node_mask,
    #                       graph_mask
    #                       )
        
    #     return dict(energy=energy, forces=forces)
    
    # return energy_and_force_fn

    # def energy_and_force_and_dipole_fn(params,
    #                         positions: jnp.ndarray,
    #                         atomic_numbers: jnp.ndarray,
    #                         idx_i: jnp.ndarray,
    #                         idx_j: jnp.ndarray,
    #                         cell: jnp.ndarray = None,
    #                         cell_offset: jnp.ndarray = None,
    #                         batch_segments: jnp.ndarray = None,
    #                         node_mask: jnp.ndarray = None,
    #                         graph_mask: jnp.ndarray = None,
    #                         *args,
    #                         **kwargs):
    #     (_, energy), forces = jax.value_and_grad(
    #         energy_fn,
    #         argnums=1,
    #         has_aux=True)(params,
    #                       positions,
    #                       atomic_numbers,
    #                       idx_i,
    #                       idx_j,
    #                       cell,
    #                       cell_offset,
    #                       batch_segments,
    #                       node_mask,
    #                       graph_mask
    #                       )

    #     if batch_segments is None:
    #         assert graph_mask is None
    #         assert node_mask is None

    #         graph_mask = jnp.ones((1,)).astype(jnp.bool_)  # (1)
    #         node_mask = jnp.ones((len(positions),)).astype(jnp.bool_)  # (num_nodes)
    #         batch_segments = jnp.zeros_like(atomic_numbers)  # (num_nodes) 

    #     inputs = dict(positions=positions,
    #             atomic_numbers=atomic_numbers,
    #             idx_i=idx_i,
    #             idx_j=idx_j,
    #             cell=cell,
    #             cell_offset=cell_offset,
    #             batch_segments=batch_segments,
    #             node_mask=node_mask,
    #             graph_mask=graph_mask
    #             )

    #     dipole = model.apply(params, inputs)['dipole']  # (num_graphs)
    #     dipole = jnp.where(graph_mask, dipole, jnp.asarray(0., dtype=dipole.dtype))  # (num_graphs)
    #     return dict(energy=energy, forces=forces, dipole=dipole)

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
                            graph_mask_expanded: jnp.ndarray = None,
                            total_charge: jnp.ndarray = None,
                            hirsh_bool: jnp.ndarray = None,
                            i_pairs: jnp.ndarray = None,
                            j_pairs: jnp.ndarray = None,
                            # d_ij_all: jnp.ndarray = None,
                            pair_mask: jnp.ndarray = None,
                            batch_segments_pairs: jnp.ndarray = None,
                            dummmy: jnp.ndarray = None,
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
                          graph_mask_expanded,
                          total_charge,
                          hirsh_bool,
                          i_pairs,
                          j_pairs,
                        #   d_ij_all,
                          pair_mask,
                          batch_segments_pairs,
                          dummmy
                          )

        if batch_segments is None:
            assert graph_mask is None
            assert graph_mask_expanded is None
            assert node_mask is None

            graph_mask = jnp.ones((1,)).astype(jnp.bool_)  # (1)
            graph_mask_expanded = jnp.ones((1,3)).astype(jnp.bool_)  # (1,3)
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
                graph_mask_expanded=graph_mask_expanded,
                total_charge=total_charge,
                hirsh_bool=hirsh_bool,
                i_pairs=i_pairs,
                j_pairs=j_pairs,
                # d_ij_all=d_ij_all,
                pair_mask=pair_mask,
                batch_segments_pairs=batch_segments_pairs,
                dummmy = dummmy
                )

        _, number_of_atoms_in_molecule = jnp.unique(batch_segments, return_counts = True, size=len(graph_mask))
        hirsh_bool_1 = jnp.repeat(hirsh_bool, number_of_atoms_in_molecule, total_repeat_length = len(node_mask))
        hirsh_bool_2 = jax.lax.convert_element_type(hirsh_bool_1, jnp.bool_)
        # print('number_of_atoms_in_molecule: ', number_of_atoms_in_molecule)
        # print('hirsh_bool: ', hirsh_bool)
        # print('hirsh_bool_1: ', hirsh_bool_1)
        # print('hirsh_bool_2: ', hirsh_bool_2)
        # # print('hirsh_bool_2: ', hirsh_bool_2.value)
        # hirsh_bool_2_values = jax.device_get(hirsh_bool_2)
        # print('hirsh_bool_2:', jnp.array(hirsh_bool_2_values))
        # print('node_mask: ', node_mask)


        dipole = model.apply(params, inputs)['dipole']  # (num_graphs)
        dipole = jnp.where(graph_mask, dipole, jnp.asarray(0., dtype=dipole.dtype))  # (num_graphs)
        dipole_vec = model.apply(params, inputs)['dipole_vec']  # (num_graphs)
        dipole_vec = safe_scale(dipole_vec, graph_mask_expanded)
        hirshfeld_ratios = model.apply(params, inputs)['hirshfeld_ratios']  # (num_graphs)
        hirshfeld_ratios = safe_scale(hirshfeld_ratios, node_mask)
        hirshfeld_ratios = safe_scale(hirsh_bool_2, hirshfeld_ratios)

        return dict(energy=energy, forces=forces, dipole=dipole, dipole_vec=dipole_vec, hirshfeld_ratios = hirshfeld_ratios)
        
    return energy_and_force_and_dipole_and_hirsh_fn