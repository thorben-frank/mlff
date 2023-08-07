import jax
import jax.numpy as jnp
import logging

from typing import Any, Callable, Tuple

from flax import struct

from mlff.utils import Graph

logging.basicConfig(level=logging.INFO)

StackNet = Any


@struct.dataclass
class CalculatorX:
    calculate_fn: Callable[[Any], jnp.ndarray] = struct.field(pytree_node=False)
    implemented_properties: Tuple = struct.field(pytree_node=False)

    @classmethod
    def create(cls, potential, implemented_properties=('energy', 'forces')):

        if 'stress' in implemented_properties:
            try:
                from glp.calculators.utils import strain_graph, get_strain
            except ImportError:
                raise ImportError('Please install GLP package for stress computation.')

            def energy_fn(atoms: Any, strain: jnp.ndarray):
                graph = atomsx_to_graph(atoms)
                graph = strain_graph(graph, strain)
                return potential(graph).sum()

            def calculate_fn(atoms: Any):
                strain = get_strain()
                energy, grads = jax.value_and_grad(energy_fn, argnums=(0, 1), allow_int=True)(atoms, strain)
                forces = - grads[0].positions
                stress = grads[1]
                return {'energy': energy, 'forces': forces, 'stress': stress}
        else:
            def energy_fn(atoms: Any):
                graph = atomsx_to_graph(atoms)
                return potential(graph).sum()

            def calculate_fn(atoms: Any):
                energy, grads = jax.value_and_grad(energy_fn, allow_int=True)(atoms)
                forces = - grads.positions
                return {'energy': energy, 'forces': forces}

        return cls(calculate_fn=calculate_fn,
                   implemented_properties=implemented_properties)

    def __call__(self, atoms: Any):
        return self.calculate_fn(atoms)
#
#
# @struct.dataclass
# class HeatFluxCalculatorX:
#     calculate_fn: Callable[[Any], jnp.ndarray] = struct.field(pytree_node=False)
#     implemented_properties: Tuple = struct.field(pytree_node=False)
#
#     @classmethod
#     def create(cls, potential, implemented_properties=('energy', 'forces', 'stress', 'heat_flux')):
#
#         def energy_fn(atoms: Any):
#             graph = atomsx_to_graph(atoms)
#             return jnp.sum(potential(graph).reshape(-1) * atoms.get_unfolding_mask())
#
#         def barycenter_fn(atoms: Any, r0: jnp.ndarray):
#             graph = atomsx_to_graph(atoms)
#             energies = potential(graph).reshape(-1) * atoms.get_unfolding_mask()
#             barycenter = energies[:, None] * r0
#             return jnp.sum(barycenter, axis=0)
#
#         def calculate_fn(atoms):
#             atoms_uf = atoms.do_unfolding()
#
#             energy, grads = jax.value_and_grad(energy_fn, argnums=0, allow_int=True)(atoms_uf)
#             forces = jax.ops.segment_sum(-grads.positions,
#                                          atoms_uf.get_unfolding_replica_idx(),
#                                          atoms.get_number_of_atoms())
#
#             stress = jnp.einsum("ia,ib->ab", atoms_uf.get_positions(), grads.positions)
#
#             velocities_uf = atoms_uf.get_velocities()
#             r0 = jax.lax.stop_gradient(atoms_uf.positions)
#
#             _, term_1 = jax.jvp(
#                 lambda R: barycenter_fn(
#                     atoms_uf.update_positions(R, update_neighbors=False), r0
#                 ),
#                 (atoms_uf.get_positions(),),
#                 (velocities_uf,),
#             )
#
#             term_2 = jnp.sum(
#                 jnp.sum(grads.positions * velocities_uf, axis=1)[:, None] * r0,
#                 axis=0,
#             )
#             heat_flux = term_1 - term_2
#
#             return {"energy": energy,
#                     "forces": forces,
#                     "stress": stress,
#                     "heat_flux": heat_flux,
#             }
#
#         return cls(implemented_properties=implemented_properties,
#                    calculate_fn=calculate_fn)
#
#     def __call__(self, atoms: Any):
#         return self.calculate_fn(atoms)


def atomsx_to_graph(atoms: Any):
    # neighbors are an *updated* neighborlist
    # question: how do we treat batching?

    positions = atoms.get_positions()
    nodes = atoms.get_atomic_numbers()

    displacement_fn = to_displacement(atoms)

    neighbors = atoms.get_neighbors()

    edges = jax.vmap(displacement_fn)(
        positions[neighbors['idx_j']], positions[neighbors['idx_i']]
    )

    mask = neighbors['idx_i'] != positions.shape[0]

    return Graph(edges, nodes, neighbors['idx_j'], neighbors['idx_i'], mask)


def to_displacement(atoms):
    from glp.periodic import make_displacement

    displacement = make_displacement(atoms.cell)

    # reverse sign convention for backwards compatibility
    return lambda Ra, Rb: raw_disp(Rb, Ra)
