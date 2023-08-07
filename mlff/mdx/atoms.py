import numpy as np
import jax
import jax.numpy as jnp

from flax import struct

from typing import Tuple, Union, Sequence

from ase.atoms import Atoms
from collections import namedtuple

from mlff.utils import Graph, Neighbors, System


SpatialPartitioning = namedtuple(
    "SpatialPartitioning", ("allocate_fn", "update_fn", "cutoff", "skin", "capacity_multiplier")
)

SpatialUnfolding = namedtuple(
    'SpatialUnfolding', ('unfolding', 'check_unfolding_fn', 'cutoff', 'skin')
)

Unfolded = namedtuple(
    'Unfolded', ('mask', 'replica_idx', 'overflow')
)


@struct.dataclass
class AtomsX:
    numbers: jnp.ndarray = struct.field(pytree_node=False)
    masses: jnp.ndarray = struct.field(pytree_node=False)
    pbc: jnp.ndarray = struct.field(pytree_node=False)

    positions: jnp.ndarray = None
    scaled_positions: jnp.ndarray = None
    momenta: jnp.ndarray = None
    magnetic_moments: jnp.ndarray = None
    charges: jnp.ndarray = None
    cell: jnp.ndarray = None

    spatial_partitioning: SpatialPartitioning = struct.field(pytree_node=False, default=None)
    neighbors: Neighbors = None

    spatial_unfolding: SpatialUnfolding = struct.field(pytree_node=False, default=None)
    unfolded: Unfolded = None

    # eckart_frame: bool = struct.field(pytree_node=False, default=False)

    @classmethod
    def create(cls,
               atoms: Atoms,
               dtype=jnp.float32):

        numbers = jnp.array(atoms.get_atomic_numbers(), dtype=jnp.int32)
        masses = jnp.array(atoms.get_masses(), dtype=dtype)
        positions = jnp.array(atoms.get_positions(), dtype=dtype)
        momenta = jnp.array(atoms.get_momenta(), dtype=dtype)
        magnetic_moments = jnp.array(atoms.get_initial_magnetic_moments(), dtype=dtype)
        charges = jnp.array(atoms.get_initial_charges(), dtype=dtype)

        # take care of different conventions. ASE assumes a cell full of zeros if no cell is set.
        if (atoms.get_cell() == 0).all():
            cell = None
        else:
            cell = jnp.array(np.array(atoms.get_cell()), dtype=dtype).T
        # note that neighborlist from glp uses different cell convention than ASE

        pbc = jnp.array(atoms.get_pbc(), dtype=dtype)

        return cls(numbers=numbers,
                   positions=positions,
                   momenta=momenta,
                   masses=masses,
                   magnetic_moments=magnetic_moments,
                   charges=charges,
                   cell=cell,
                   pbc=pbc,
                   )

    def init_spatial_unfolding(self, cutoff, skin):
        if self.spatial_partitioning is not None:
            raise RuntimeError('First initialize the spatial unfolding then the spatial partitioning.')

        try:
            from glp.system import unfold_system
            from glp.unfold import unfolder
        except ImportError:
            raise ImportError('For spatial unfolding, please install the glp package from ...')

        unfolding, check_unfolding = unfolder(self.to_system(), cutoff, skin)
        uf_system = unfold_system(self.to_system(), unfolding)
        unfolded = Unfolded(mask=uf_system.mask,
                            replica_idx=uf_system.replica_idx,
                            overflow=check_unfolding(self.to_system(),
                                                     unfolding))

        return self.replace(spatial_unfolding=SpatialUnfolding(check_unfolding_fn=check_unfolding,
                                                               unfolding=unfolding,
                                                               cutoff=cutoff,
                                                               skin=skin
                                                               ),
                            unfolded=unfolded
                            )

    def init_spatial_partitioning(self, cutoff, skin, capacity_multiplier=1.25):
        if self.spatial_unfolding is not None:
            try:
                from glp.system import unfold_system
            except ImportError:
                raise ImportError('For spatial unfolding, please install the glp package from ...')
            system = unfold_system(self.to_system(), self.spatial_unfolding.unfolding)
        else:
            try:
                from glp.neighborlist import quadratic_neighbor_list
            except ImportError:
                raise ImportError('For neighborhood list, please install the glp package from ...')
            system = self.to_system()

        neighbors, spatial_partitioning = neighbor_list(system,
                                                        cutoff=cutoff,
                                                        skin=skin,
                                                        capacity_multiplier=capacity_multiplier)

        return self.replace(spatial_partitioning=spatial_partitioning,
                            neighbors=neighbors)

    def reset_spatial_partitioning(self):
        return self.replace(spatial_partitioning=None,
                            neighbors=None)

    def update_neighbors(self):
        if self.spatial_partitioning is None:
            raise RuntimeError('AtomsX object has no spatial partitioning.')

        new_neighbors = self.spatial_partitioning.update_fn(self, self.neighbors)
        return self.replace(neighbors=new_neighbors)

    def get_neighbors(self):
        if self.spatial_partitioning is None:
            raise RuntimeError('AtomsX object has no spatial partitioning.')

        return {'idx_i': self.neighbors.centers, 'idx_j': self.neighbors.others}

    def do_unfolding(self):
        if self.spatial_unfolding is None:
            raise RuntimeError('AtomsX object has no spatial unfolding.')

        try:
            from glp.system import unfold_system
        except ImportError:
            raise ImportError('For spatial unfolding, please install the glp package from ...')

        uf_system = unfold_system(self.to_system(), self.spatial_unfolding.unfolding)
        unfolded = Unfolded(mask=uf_system.mask,
                            replica_idx=uf_system.replica_idx,
                            overflow=self.spatial_unfolding.check_unfolding_fn(self.to_system(),
                                                                               self.spatial_unfolding.unfolding))
        return self.replace(positions=uf_system.R,
                            numbers=uf_system.Z,
                            momenta=self.get_momenta()[unfolded.replica_idx],
                            masses=self.get_masses()[unfolded.replica_idx],
                            unfolded=unfolded)

    def get_unfolding_mask(self):
        if self.spatial_unfolding is None:
            raise RuntimeError('AtomsX object has no spatial unfolding.')
        return self.unfolded.mask

    def get_unfolding_replica_idx(self):
        if self.spatial_unfolding is None:
            raise RuntimeError('AtomsX object has no spatial unfolding.')
        return self.unfolded.replica_idx

    def update_velocities(self, velocities):
        """
        Update the momenta by specifying the velocities.

        Args:
            velocities (Array):

        Returns:

        """
        assert velocities.shape[-1] == 3
        assert velocities.shape[0] == self.get_number_of_atoms()
        assert velocities.ndim == 2

        return self.replace(momenta=(self.get_masses()[:, None] * velocities))

    def update_momenta(self, momenta):
        """
        Update the momenta.

        Args:
            momenta (Array): Momenta, shape: (n,3)

        Returns: Updated AtomsX.

        """
        return self.replace(momenta=momenta)

    def get_kinetic_energy(self):
        """
        Get the kinetic energy of the system, given the momenta.

        Returns: The kinetic energy.

        """
        return 0.5 * jnp.vdot(self.momenta, self.get_velocities())

    def get_temperature(self, eckart_frame: bool = True):
        """
        Get the temperature of the system, calculated from the kinetic energy. See e.g.

        Args:
            eckart_frame (bool): Is the atoms in the eckart frame. If True, DOF = 3*n - 6. If False DOF = 3*n

        Returns: Temperature in eV.

        """
        if eckart_frame:
            dof = 3 * self.get_number_of_atoms() - 6
        else:
            dof = 3 * self.get_number_of_atoms()

        return 2*self.get_kinetic_energy() / dof

    def get_velocities(self) -> jnp.ndarray:
        """
        Get the velocity per atom.

        Returns: Velocities, shape: (n,3)

        """
        momenta = self.get_momenta()
        masses = self.get_masses()
        return momenta / masses[:, None]

    def get_momenta(self) -> jnp.ndarray:
        """
        Get the momenta per atom.

        Returns: Momenta, shape: (n,3)

        """
        return self.momenta

    def get_moments_of_inertia(self, vectors=False) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Get the moments of inertia along the principal axes.

        The three principal moments of inertia are computed from the
        eigenvalues of the symmetric inertial tensor. Periodic boundary
        conditions are ignored. Units of the moments of inertia are
        amu*angstrom**2.

        Args:
            vectors (bool): Return the basis vectors.

        Returns: (Moments, Optional[Basis])

        """

        com = self.get_center_of_mass()
        positions = self.get_positions()
        positions -= com
        masses = self.get_masses()

        @jax.vmap
        def calculate_I(pos: jnp.ndarray, m: jnp.ndarray):
            x, y, z = pos
            I11 = m * (y ** 2 + z ** 2)
            I22 = m * (x ** 2 + z ** 2)
            I33 = m * (x ** 2 + y ** 2)
            I12 = -m * x * y
            I13 = -m * x * z
            I23 = -m * y * z

            return jnp.array([[I11, I12, I13],
                              [I12, I22, I23],
                              [I13, I23, I33]])

        I = calculate_I(positions, masses).sum(0)
        evals, evecs = jnp.linalg.eigh(I)

        if vectors:
            return evals, evecs.transpose()
        else:
            return evals

    def get_atomic_numbers(self) -> jnp.ndarray:
        """
        Get the atomic numbers.

        Returns: Array, shape: (n)

        """
        return self.numbers

    def get_masses(self) -> jnp.ndarray:
        """
        Get the atomic masses.

        Returns: Array, shape: (n)

        """
        return self.masses

    def get_positions(self) -> jnp.ndarray:
        """
        Get the atomic positions.

        Returns: Array, shape: (n,3)

        """
        return self.positions

    def get_initial_magnetic_moments(self) -> jnp.ndarray:
        """
        Get initial magnetic moments.

        Returns: Array, shape:

        """
        return self.magnetic_moments

    def get_initial_charges(self) -> jnp.ndarray:
        """
        Get the initial charges.

        Returns: Array, shape: (n,1)

        """
        return self.charges

    def get_cell(self) -> Union[jnp.ndarray, None]:
        """
        Get the super cell. The cell is a 3x3 `class:jnp.ndarray`, where cell[i, j] is the i-th coordinate of the
        j-th cell vector.
        If no cell is set, return `None`.

        Returns: Array, shape: (3,3)

        """
        return self.cell

    def get_pbc(self) -> jnp.ndarray:
        """
        Get periodic boundary conditions.

        Returns: Array, shape: (3)

        """
        return self.pbc

    def get_number_of_atoms(self) -> int:
        """
        Get the total number of atoms.

        Returns: int

        """
        return len(self.numbers)

    def update_positions(self, positions, update_neighbors: bool = True):
        if update_neighbors:
            if self.spatial_partitioning is None:
                raise RuntimeError('AtomsX object has no spatial partitioning.')

            if self.spatial_unfolding is not None:
                _atoms = self.update_positions(positions=positions, update_neighbors=False)
                _atoms = _atoms.do_unfolding()
                _unfolded = _atoms.unfolded
            else:
                _atoms = self.update_positions(positions=positions, update_neighbors=False)
                _unfolded = None

            new_neighbors = self.spatial_partitioning.update_fn(_atoms, self.neighbors)
            return self.replace(positions=positions,
                                neighbors=new_neighbors,
                                unfolded=_unfolded)
        else:
            return self.replace(positions=positions)

    def get_center_of_mass(self, scaled=False):
        """
        Calculate and returns center of mass for current atomic positions. If scaled=True the center of mass
        in scaled coordinates is returned.
        Args:
            scaled (bool): Center of mass in scaled coordinates.

        Returns:

        """
        masses = self.get_masses()
        com = masses @ self.get_positions() / masses.sum()
        if scaled:
            raise NotImplementedError("get_center_of_mass() currently not implemented for scaled=True")
        else:
            return com

    def update_center_of_mass(self, com, scaled=False):
        """Return the atomic coordinates with an updated center of mass. If scaled=True the center of mass is expected
        in scaled coordinates.

        Args:
            scaled (bool): Center of mass in scaled coordinates.

        Returns:

        """
        old_com = self.get_center_of_mass(scaled=scaled)
        difference = old_com - com
        if scaled:
            raise NotImplementedError('update_center_of_mass() currently not implemented for scaled=True')
            # self.set_scaled_positions(self.get_scaled_positions() + difference)
        else:
            return self.update_positions(self.get_positions() + difference)

    def get_center_of_mass_velocity(self):
        """
        Get velocity of the center of mass.

        Returns: Velocity of the center of mass, shape: (3)

        """

        return jnp.dot(self.get_masses().ravel(), self.get_velocities()) / self.get_masses().sum()

    def get_angular_momentum(self):
        """
        Get total angular momentum with respect to the center of mass.

        Returns:

        """

        com = self.get_center_of_mass()
        positions = self.get_positions()
        positions -= com
        return jnp.cross(positions, self.get_momenta()).sum(0)

    def to_system(self):
        """
        Transform the atomsX object to a glp.system.

        Returns: glp.System

        """
        pos = self.get_positions()
        z = self.get_atomic_numbers()
        cell = self.get_cell()
        return System(R=pos, Z=z, cell=cell)

    def to_graph(self):
        """
        Transform the atomsX object to a glp.graph.

        Returns: glp.graph

        """
        positions = self.get_positions()
        nodes = self.get_atomic_numbers()

        displacement_fn = to_displacement(self)

        neighbors = self.get_neighbors()

        edges = jax.vmap(displacement_fn)(
            positions[neighbors['idx_j']], positions[neighbors['idx_i']]
        )

        mask = neighbors['idx_i'] != positions.shape[0]

        return Graph(edges, nodes, neighbors['idx_i'], neighbors['idx_j'], mask)

    def update_cell(self, cell):
        return self.replace(cell=cell)

    def rotate(self, angles: Sequence[int], euler_axes: str = 'xyz', degrees: bool = True):
        from mlff.geometric import rotate_by
        pos = self.get_positions()
        pos_rot = rotate_by(pos, angles=angles, euler_axes=euler_axes, degrees=degrees)
        return self.update_positions(pos_rot, update_neighbors=False)


def neighbor_list(system, cutoff, skin, capacity_multiplier=1.25):
    try:
        from glp.neighborlist import quadratic_neighbor_list
    except ImportError:
        raise ImportError('For neighborhood list, please install the glp package from ...')
    # Convenience interface for system but with for atomsX adapted update_fn

    allocate, update = quadratic_neighbor_list(
        system.cell, cutoff, skin, capacity_multiplier=capacity_multiplier
    )

    def _update(x: AtomsX, neighbors):
        system = x.to_system()
        neighbors = update(system.R, neighbors, new_cell=system.cell)
        return neighbors

    neighbors = allocate(system.R)

    return neighbors, SpatialPartitioning(allocate_fn=allocate,
                                          update_fn=_update,
                                          cutoff=cutoff,
                                          skin=skin,
                                          capacity_multiplier=capacity_multiplier)


def to_displacement(atoms):
    from glp.periodic import make_displacement

    displacement = make_displacement(atoms.cell)

    # reverse sign convention for backwards compatibility
    return lambda Ra, Rb: raw_disp(Rb, Ra)
