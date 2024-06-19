import jax
import jax.numpy as jnp
import numpy as np
import logging

from collections import namedtuple
from typing import Any

from ase.calculators.calculator import Calculator

from mlff.utils.structures import Graph
from mlff.mdx.potential import MLFFPotentialSparse
try:
    from glp.calculators.utils import strain_graph, get_strain, strain_system
    from glp import System, atoms_to_system
    from glp.graph import system_to_graph
except ImportError:
    raise ImportError('Please install GLP package for running MD.')

SpatialPartitioning = namedtuple(
    "SpatialPartitioning", ("allocate_fn", "update_fn", "cutoff", "lr_cutoff", "skin", "capacity_multiplier", "buffer_size_multiplier")
)

PME = namedtuple("PME", ("ngrid", "alpha", "tolerance", "frequencies"))
logging.basicConfig(level=logging.INFO)

StackNet = Any

def matrix_to_voigt(matrix):
    """
    Convert a 3x3 matrix to a 6-component stress vector in Voigt notation.

    Args:
        matrix (jnp.ndarray): A 3x3 matrix.

    Returns:
        jnp.ndarray: A 6-component stress vector in Voigt notation.
    """
    ## Check input
    if matrix.shape != (3, 3):
        raise ValueError("Input must be a 3x3 matrix. Shape is ", matrix.shape)

    ## Form Voigt vector
    voigt_vector = jnp.array([matrix[0,0], matrix[1,1], matrix[2,2], (matrix[1,2]+matrix[2,1])/2, (matrix[0,2]+matrix[2,0])/2, (matrix[0,1]+matrix[1,0])/2])

    return voigt_vector

class mlffCalculatorSparse(Calculator):
    implemented_properties = ['energy', 'forces', 'stress', 'free_energy']

    @classmethod
    def create_from_ckpt_dir(cls,
                             ckpt_dir: str,
                             calculate_stress: bool = False,
                             use_PME: bool = False,
                             lr_cutoff: float = 10.,
                             E_to_eV: float = 1.,
                             F_to_eV_Ang: float = 1.,
                             capacity_multiplier: float = 1.25,
                             buffer_size_multiplier: float = 1.25,
                             skin: float = 0.,
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
                   buffer_size_multiplier=buffer_size_multiplier,
                   skin=skin,
                   use_PME=use_PME,
                   lr_cutoff=lr_cutoff,
                   dtype=dtype,
                   has_aux=has_aux
                   )

    def __init__(
            self,
            potential,
            E_to_eV: float = 1.,
            F_to_eV_Ang: float = 1.,
            capacity_multiplier: float = 1.25,
            buffer_size_multiplier: float = 1.25,
            skin: float = 0.,
            calculate_stress: bool = False,
            use_PME: bool = False,
            lr_cutoff: float = 10.,
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
            def energy_fn(system, strain: jnp.ndarray, neighbors, pme):
                strained_system = strain_system(system, strain)
                graph = system_to_graph(strained_system, neighbors, pme)

                #graph = system_to_graph(system, neighbors, pme)
                #graph = strain_graph(graph, strain)

                out = potential(graph, has_aux=has_aux)
                if isinstance(out, tuple):
                    atomic_energy = out[0]
                    aux = out[1]
                    return atomic_energy.sum(), aux
                else:
                    atomic_energy = out
                    return atomic_energy.sum()

            @jax.jit
            def calculate_fn(system: System, neighbors, pme):
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
                    pme
                  )

                forces = - grads[0].R
                volume = jnp.abs(jnp.dot(jnp.cross(system.cell[0], system.cell[1]), system.cell[2]))
                stress = grads[1] / volume
                stress = matrix_to_voigt(stress)

                if isinstance(out, tuple):
                    if not has_aux:
                        raise ValueError

                    return {'energy': out[0], 'forces': forces, 'stress': stress, 'aux': out[1]}
                else:
                    return {'energy': out, 'forces': forces, 'stress': stress}

        else:
            def energy_fn(system, neighbors, pme):
                graph = system_to_graph(system, neighbors, pme)
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
            def calculate_fn(system, neighbors, pme):
                out, grads = jax.value_and_grad(
                    energy_fn,
                    allow_int=True,
                    has_aux=has_aux
                )(
                    system,
                    neighbors,
                    pme
                )
                forces = - grads.R

                if isinstance(out, tuple):
                    if not has_aux:
                        raise ValueError

                    return {'energy': out[0], 'forces': forces, 'aux': out[1]}
                else:
                    return {'energy': out, 'forces': forces}

        self.calculate_fn = calculate_fn
        self.use_PME = use_PME #boolean flag
        self.pme = None #Tuple of ngrid, alpha, tolerance, frequencies
        self.neighbors = None
        self.spatial_partitioning = None
        self.capacity_multiplier = capacity_multiplier
        self.buffer_size_multiplier = buffer_size_multiplier
        self.skin = skin
        self.cutoff = potential.cutoff # cutoff for the neighbor list
        self.lr_cutoff = lr_cutoff # cutoff for electrostatics
        self.dtype = dtype

    def calculate(self, atoms=None, *args, **kwargs):
        super(mlffCalculatorSparse, self).calculate(atoms, *args, **kwargs)

        system = atoms_to_system(atoms)

        if atoms.get_pbc().any():
            cell = jnp.array(np.array(atoms.get_cell()), dtype=self.dtype).T  # (3,3)
        else:
            cell = None
        # Allocate the grid for PME. It might be necessary to put some tollerance in the cell
        # to fullfil that the grid is big enough to keep the max distance between points    
        if self.pme is None and self.use_PME:
            self.pme = get_ngrid(cell, self.lr_cutoff, tolerance=5e-4)
            
        if self.spatial_partitioning is None:
            self.neighbors, self.spatial_partitioning = neighbor_list(positions=system.R,
                                                                      cell=cell,
                                                                      cutoff=self.cutoff,
                                                                      skin=self.skin,
                                                                      capacity_multiplier=self.capacity_multiplier,
                                                                      buffer_size_multiplier=self.buffer_size_multiplier, 
                                                                      lr_cutoff=self.lr_cutoff)
        neighbors = self.spatial_partitioning.update_fn(system.R, self.neighbors, new_cell=cell)
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
                                                                      skin=self.skin,
                                                                      capacity_multiplier=self.capacity_multiplier,
                                                                      buffer_size_multiplier=self.buffer_size_multiplier,
                                                                      lr_cutoff=self.lr_cutoff)
            #self.neighbors now contains Neighbors namedtuple with idx_i_lr etc

        output = self.calculate_fn(system, neighbors, self.pme)  # note different cell convention
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

def get_ngrid(cell, r_cut, tolerance):
    """ Get the grid for PME calculation. Alpha is calculated from the tolerance and r_cut.
    Args:
        cell (np.ndarray): 3x3 matrix of cell vectors.
        r_cut (float): cutoff radius.
        tolerance (float): tolerance for the PME calculation. Usually 5e-4.
    Returns:
        PME: PME object containing the grid, alpha, tolerance and the frequencies.
    """
    alpha = np.sqrt(-np.log(2 * tolerance)) / r_cut
    lcell = np.linalg.norm(cell, axis=0)
    ngrid = np.ceil((2 * alpha * lcell) / (3 * tolerance ** (1 / 5))).astype(int)
    freq = jnp.meshgrid(*[jnp.fft.fftfreq(g) for g in ngrid], indexing='ij')
    ngrid = jnp.zeros((*ngrid,), dtype=jnp.float32)
    return PME(ngrid, alpha, tolerance, freq)

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


def neighbor_list(positions: jnp.ndarray, cutoff: float, skin: float = 0., cell: jnp.ndarray = None,
        capacity_multiplier: float = 1.25, lr_cutoff: float = 10., buffer_size_multiplier: float = 1.25):
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
        cell, cutoff, skin, capacity_multiplier=capacity_multiplier, use_cell_list=True, lr_cutoff=lr_cutoff, buffer_size_multiplier=buffer_size_multiplier
    )
    neighbors = allocate(positions)
    return neighbors, SpatialPartitioning(allocate_fn=allocate,
                                          update_fn=jax.jit(update),
                                          cutoff=cutoff,
                                          skin=skin,
                                          capacity_multiplier=capacity_multiplier,
                                          buffer_size_multiplier=buffer_size_multiplier, 
                                          lr_cutoff=lr_cutoff)
