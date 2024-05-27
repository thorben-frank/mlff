import jax
import jax.numpy as jnp
import numpy as np
import logging

from collections import namedtuple
from typing import Any

from ase.calculators.calculator import Calculator
from ase.neighborlist import neighbor_list as ase_neighbor_list

from mlff.utils.structures import Graph
from mlff.mdx.potential import MLFFPotentialSparse
import jax.profiler
try:
    from glp.calculators.utils import strain_graph, get_strain
    from glp import System, atoms_to_system
    from glp.graph import system_to_graph
except ImportError:
    raise ImportError('Please install GLP package for running MD.')

SpatialPartitioning = namedtuple(
    "SpatialPartitioning", ("allocate_fn", "update_fn", "cutoff", "skin", "capacity_multiplier")
)
Pairs = namedtuple("Pairs", ("idx_i_lr", "idx_j_lr"))

logging.basicConfig(level=logging.INFO)

StackNet = Any


class mlffCalculatorSparse(Calculator):
    implemented_properties = ['energy', 'forces', 'stress', 'free_energy']

    @classmethod
    def create_from_ckpt_dir(cls,
                             ckpt_dir: str,
                             calculate_stress: bool = False,
                             E_to_eV: float = 1.,
                             F_to_eV_Ang: float = 1.,
                             capacity_multiplier: float = 1.75,
                             add_energy_shift: bool = False,
                             dtype: np.dtype = np.float32,
                             model: str = 'so3krates',
                             has_aux: bool = False):

        mlff_potential = MLFFPotentialSparse.create_from_ckpt_dir(
            ckpt_dir=ckpt_dir,
            add_shift=add_energy_shift,
            dtype=dtype,
            model=model,
        )

        return cls(potential=mlff_potential,
                   calculate_stress=calculate_stress,
                   E_to_eV=E_to_eV,
                   F_to_eV_Ang=F_to_eV_Ang,
                   capacity_multiplier=capacity_multiplier,
                   dtype=dtype,
                   has_aux=has_aux
                   )

    def __init__(
            self,
            potential,
            E_to_eV: float = 1.,
            F_to_eV_Ang: float = 1.,
            capacity_multiplier: float = 1.75,
            calculate_stress: bool = False,
            dtype: np.dtype = np.float32,
            has_aux: bool = False,
            *args,
            **kwargs
    ):
        """
        ASE calculator given a StackNet and parameters.

        A calculator takes atomic numbers and atomic positions from an Atoms object and calculates the energy and
        forces.

        Args:
            E_to_eV (float): Conversion factor from whatever energy unit is used by the model to eV.
                By default this parameter is set to convert from kcal/mol.
            F_to_eV_Ang (float): Conversion factor from whatever length unit is used by the model to Angstrom. By
                default, the length unit is not converted (assumed to be in Angstrom)
            *args ():
            **kwargs ():
        """
        super(mlffCalculatorSparse, self).__init__(*args, **kwargs)
        self.log = logging.getLogger(__name__)
        self.log.warning(
            'Please remember to specify the proper conversion factors, if your model does not use '
            '\'eV\' and \'Ang\' as units.'
        )
        if calculate_stress:
            def energy_fn(system, strain: jnp.ndarray, neighbors, pairs):
                graph = system_to_graph(system, neighbors, pairs)
                graph = strain_graph(graph, strain)

                out, aux = potential(graph, has_aux=has_aux)
                if isinstance(out, tuple):
                    atomic_energy = out[0]
                    aux = out[1]
                    return atomic_energy.sum(), aux
                else:
                    atomic_energy = out
                    return atomic_energy.sum()

            @jax.jit
            def calculate_fn(system: System, neighbors, pairs):
                strain = get_strain()
                out, grads = jax.value_and_grad(
                    energy_fn,
                    argnums=(0, 1),
                    allow_int=True,
                    has_aux=has_aux
                )(
                    system,
                    strain,
                    neighbors,
                    pairs
                  )

                forces = - grads[0].R
                volume_factor = jnp.abs(jnp.dot(jnp.cross(system.cell[0], system.cell[1]), system.cell[2]))
                stress = grads[1] / volume_factor

                if isinstance(out, tuple):
                    if not has_aux:
                        raise ValueError

                    return {'energy': out[0], 'forces': forces, 'stress': stress, 'aux': out[1]}
                else:
                    return {'energy': out, 'forces': forces, 'stress': stress}

        else:
            def energy_fn(system, neighbors, pairs):
                graph = system_to_graph(system, neighbors, pairs)
                out = potential(graph, has_aux=has_aux)
                if isinstance(out, tuple):
                    if not has_aux:
                        raise ValueError

                    atomic_energy = out[0]
                    aux = out[1]
                    return atomic_energy.sum(), aux
                else:
                    atomic_energy = out
                    return atomic_energy.sum()

            @jax.jit
            def calculate_fn(system, neighbors,  pairs):
                out, grads = jax.value_and_grad(
                    energy_fn,
                    allow_int=True,
                    has_aux=has_aux
                )(
                    system,
                    neighbors,
                    pairs
                )
                forces = - grads.R

                if isinstance(out, tuple):
                    if not has_aux:
                        raise ValueError

                    return {'energy': out[0], 'forces': forces, 'aux': out[1]}
                else:
                    return {'energy': out, 'forces': forces}

        self.calculate_fn = calculate_fn

        self.neighbors = None
        self.spatial_partitioning = None
        self.capacity_multiplier = capacity_multiplier
        self.pairs = None
        self.cutoff = potential.cutoff

        self.dtype = dtype

    def calculate(self, atoms=None, *args, **kwargs):
        super(mlffCalculatorSparse, self).calculate(atoms, *args, **kwargs)

        system = atoms_to_system(atoms)

        if atoms.get_pbc().any():
            cell = jnp.array(np.array(atoms.get_cell()), dtype=self.dtype).T  # (3,3)
        else:
            cell = None

        if self.spatial_partitioning is None:
            self.neighbors, self.spatial_partitioning = neighbor_list(positions=system.R,
                                                                      cell=cell,
                                                                      cutoff=self.cutoff,
                                                                      skin=0.,
                                                                      capacity_multiplier=self.capacity_multiplier)
        neighbors = self.spatial_partitioning.update_fn(system.R, self.neighbors, cell)
        if neighbors.overflow:
            raise RuntimeError('Spatial overflow.')
        else:
            self.neighbors = neighbors
        if neighbors.cell_list is not None:
            # If cell list needs to be reallocated, then reallocate neighbors
            if neighbors.cell_list.reallocate:
                self.neighbors, self.spatial_partitioning = neighbor_list(positions=system.R,
                                                                      cell=cell,
                                                                      cutoff=self.cutoff,
                                                                      skin=0.,
                                                                      capacity_multiplier=self.capacity_multiplier)
        if self.pairs is None:
            idx_i_lrr, idx_j_lrr = ase_neighbor_list('ij', atoms, 100, self_interaction=False)
#            idx_i_lrr, idx_j_lrr = ase_neighbor_list('ij', atoms, 10, self_interaction=False)
            self.pairs = Pairs(jnp.array(idx_i_lrr, dtype=jnp.int32), jnp.array(idx_j_lrr, dtype=jnp.int32))

        output = self.calculate_fn(system, neighbors, self.pairs)  # note different cell convention
        self.results = jax.tree_map(lambda x: np.array(x), output)

def to_displacement(cell):
    """
    Returns function to calculate replacement. Returned function takes Ra and Rb as input and return Ra - Rb

    Args:
        cell ():

    Returns:

    """
    from glp.periodic import make_displacement

    displacement = make_displacement(cell)
    # displacement(Ra, Rb) calculates Rb - Ra

    # reverse sign convention bc feels more natural
    return lambda Ra, Rb: displacement(Rb, Ra)


@jax.jit
def to_graph(atomic_numbers, positions, cell, neighbors):
    """
    Transform the atomsX object to a glp.graph.

    Returns: glp.graph

    """

    displacement_fn = to_displacement(cell)
    # displacement_fn(Ra, Rb) calculates Ra - Rb

    edges = jax.vmap(displacement_fn)(
        positions[neighbors.others], positions[neighbors.centers]
    )

    mask = neighbors.centers != positions.shape[0]

    return Graph(edges=edges, nodes=atomic_numbers, centers=neighbors.centers, others=neighbors.others, mask=mask)


@jax.jit
def add_batch_dim(tree):
    return jax.tree_map(lambda x: x[None], tree)


def neighbor_list(positions: jnp.ndarray, cutoff: float, skin: float, cell: jnp.ndarray = None,
                  capacity_multiplier: float = 1.75):
    """

    Args:
        positions ():
        cutoff ():
        skin ():
        cell (): ASE cell.
        capacity_multiplier ():

    Returns:

    """
    try:
        from glp.neighborlist import quadratic_neighbor_list
    except ImportError:
        raise ImportError('For neighborhood list, please install the glp package from ...')

    allocate, update = quadratic_neighbor_list(
        cell, cutoff, skin, capacity_multiplier=capacity_multiplier, use_cell_list=True
    )
    neighbors = allocate(positions)
    return neighbors, SpatialPartitioning(allocate_fn=allocate,
                                          update_fn=jax.jit(update),
                                          cutoff=cutoff,
                                          skin=skin,
                                          capacity_multiplier=capacity_multiplier)
