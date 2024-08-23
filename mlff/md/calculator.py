import jax
import jax.numpy as jnp
import numpy as np
import logging

from collections import namedtuple
from typing import Any

from ase.calculators.calculator import Calculator

from mlff.utils.structures import Graph
from mlff.mdx.potential import MLFFPotential
try:
    from glp.calculators.utils import strain_graph, get_strain
    from glp import System, atoms_to_system
    from glp.graph import system_to_graph
except ImportError:
    raise ImportError('Please install GLP package for running MD.')

SpatialPartitioning = namedtuple(
    "SpatialPartitioning", ("allocate_fn", "update_fn", "cutoff", "skin", "capacity_multiplier")
)

logging.basicConfig(level=logging.INFO)

StackNet = Any


class mlffCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'stress', 'free_energy']

    @classmethod
    def create_from_ckpt_dir(cls,
                             ckpt_dir: str,
                             calculate_stress: bool = False,
                             E_to_eV: float = 1.,
                             F_to_eV_Ang: float = 1.,
                             capacity_multiplier: float = 1.25,
                             add_energy_shift: bool = False,
                             dtype: np.dtype = np.float64):

        mlff_potential = MLFFPotential.create_from_ckpt_dir(
            ckpt_dir=ckpt_dir,
            add_shift=add_energy_shift,
            dtype=dtype
        )

        return cls(potential=mlff_potential,
                   calculate_stress=calculate_stress,
                   E_to_eV=E_to_eV,
                   F_to_eV_Ang=F_to_eV_Ang,
                   capacity_multiplier=capacity_multiplier,
                   dtype=dtype,
                   )

    def __init__(
            self,
            potential,
            E_to_eV: float = 1.,
            F_to_eV_Ang: float = 1.,
            capacity_multiplier: float = 1.25,
            calculate_stress: bool = False,
            dtype: np.dtype = np.float64,
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
        super(mlffCalculator, self).__init__(*args, **kwargs)
        self.log = logging.getLogger(__name__)
        self.log.warning(
            'Please remember to specify the proper conversion factors, if your model does not use '
            '\'eV\' and \'Ang\' as units.'
        )
        if calculate_stress:
            def energy_fn(system, strain: jnp.ndarray, neighbors):
                graph = system_to_graph(system, neighbors)
                graph = strain_graph(graph, strain)
                return potential(graph).sum()

            @jax.jit
            def calculate_fn(system: System, neighbors):
                strain = get_strain()
                energy, grads = jax.value_and_grad(energy_fn, argnums=(0, 1), allow_int=True)(system, strain, neighbors)
                forces = - grads[0].R
                stress = grads[1]
                return {'energy': energy, 'forces': forces, 'stress': stress}
        else:
            def energy_fn(system, neighbors):
                graph = system_to_graph(system, neighbors)
                return potential(graph).sum()

            @jax.jit
            def calculate_fn(system, neighbors):
                energy, grads = jax.value_and_grad(energy_fn, allow_int=True)(system, neighbors)
                forces = - grads.R
                return {'energy': energy, 'forces': forces}

        self.calculate_fn = calculate_fn

        self.neighbors = None
        self.spatial_partitioning = None
        self.capacity_multiplier = capacity_multiplier

        self.cutoff = potential.cutoff

        self.dtype = dtype

    def calculate(self, atoms=None, *args, **kwargs):
        super(mlffCalculator, self).calculate(atoms, *args, **kwargs)

        R = jnp.array(atoms.get_positions(), dtype=self.dtype)  # shape: (n,3)
        z = jnp.array(atoms.get_atomic_numbers(), dtype=jnp.int16)  # shape: (n)

        if atoms.get_pbc().any():
            cell = jnp.array(np.array(atoms.get_cell()), dtype=self.dtype).T  # (3,3)
        else:
            cell = None

        if self.spatial_partitioning is None:
            self.neighbors, self.spatial_partitioning = neighbor_list(positions=R,
                                                                      cell=cell,
                                                                      cutoff=self.cutoff,
                                                                      skin=0.,
                                                                      capacity_multiplier=self.capacity_multiplier)

        neighbors = self.spatial_partitioning.update_fn(R, self.neighbors)
        if neighbors.overflow:
            raise RuntimeError('Spatial overflow.')
        else:
            self.neighbors = neighbors

        output = self.calculate_fn(System(R=R, Z=z, cell=cell), neighbors=neighbors)  # note different cell convention

        self.results = jax.tree_map(lambda x: np.array(x, dtype=self.dtype), output)


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


@jax.jit
def apply_neighbor_convention(tree):
    idx_i = jnp.where(tree['idx_i'] < len(tree['z']), tree['idx_i'], -1)
    idx_j = jnp.where(tree['idx_j'] < len(tree['z']), tree['idx_j'], -1)
    tree['idx_i'] = idx_i
    tree['idx_j'] = idx_j
    return tree


def neighbor_list(positions: jnp.ndarray, cutoff: float, skin: float, cell: jnp.ndarray = None,
                  capacity_multiplier: float = 1.25):
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
    # Convenience interface
    # if cell is not None:
    #     cell_T = cell.T
    # else:
    #     cell_T = None

    allocate, update = quadratic_neighbor_list(
        cell, cutoff, skin, capacity_multiplier=capacity_multiplier
    )

    neighbors = allocate(positions)

    return neighbors, SpatialPartitioning(allocate_fn=allocate,
                                          update_fn=jax.jit(update),
                                          cutoff=cutoff,
                                          skin=skin,
                                          capacity_multiplier=capacity_multiplier)


def make_phonon_calculator(ckpt_dir, n_replicas, **kwargs):
    from glp.vibes import calculator

    vibes_calc = calculator(potential={'potential': 'mlff', 'ckpt_dir': ckpt_dir},
                            calculate={'calculate': 'supercell',
                                       'n_replicas': n_replicas},
                            **kwargs
                            )
    return vibes_calc
