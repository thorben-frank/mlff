import numpy as np
import jax.numpy as jnp
from ase.io import iread
from ase.neighborlist import neighbor_list
from ase import Atoms
from dataclasses import dataclass
from typing import Dict, List
from tqdm import tqdm
import jraph

from mlff.padding import (pad_forces,
                          pad_atomic_types,
                          pad_coordinates)


@dataclass
class AseDataLoader:
    """ASE data loader.

    Attributes:
        input_file:
        output_file:
        load_stress:
        load_energy_and_forces:
        neighbors_format: dense or sparse
    """
    input_file: str
    output_file: str = None
    load_stress: bool = False
    load_energy_and_forces: bool = True
    neighbors_format: str = 'dense'

    def _load_all_dense(self) -> Dict:
        def extract_positions(x: Atoms):
            return x.get_positions()

        def extract_numbers(x: Atoms):
            return x.get_atomic_numbers()

        def extract_energy(x: Atoms):
            return x.get_potential_energy()

        def extract_forces(x: Atoms):
            return x.get_forces()

        def extract_stress(x: Atoms):
            return x.get_stress(voigt=False)

        def extract_pbc(x: Atoms):
            return x.get_pbc()

        def extract_unit_cell(x: Atoms):
            return np.array(x.get_cell(complete=False))

        pos = []
        nums = []

        energies = []
        forces = []
        stress = []

        cell = []
        pbc = []

        n_max = max(set(map(lambda x: len(x.get_atomic_numbers()), iread(self.input_file))))

        print(f"Read data from {self.input_file} ...")
        for a in tqdm(iread(self.input_file)):
            pos += [pad_coordinates(extract_positions(a)[None], n_max=n_max).squeeze(axis=0)]
            nums += [pad_atomic_types(extract_numbers(a)[None], n_max=n_max).squeeze(axis=0)]
            cell += [extract_unit_cell(a)]
            pbc += [extract_pbc(a)]

            if self.load_energy_and_forces:
                energies += [extract_energy(a)]
                forces += [pad_forces(extract_forces(a)[None], n_max=n_max).squeeze(axis=0)]

            if self.load_stress:
                stress += [extract_stress(a)]

        loaded_data = {'R': np.stack(pos, axis=0),
                       'z': np.stack(nums, axis=0),
                       'pbc': np.stack(pbc, axis=0),
                       'unit_cell': np.stack(cell, axis=0)
                       }
        if self.load_stress:
            loaded_data.update({'stress': np.stack(stress, axis=0)})
        if self.load_energy_and_forces:
            loaded_data.update({'E': np.stack(energies, axis=0).reshape(-1, 1),
                                'F': np.stack(forces, axis=0)})

        node_mask = np.where(loaded_data['z'] > 0, True, False)
        loaded_data.update({'node_mask': node_mask})

        print("... done!")
        if self.output_file is not None:
            print(f'Write data from {self.input_file} to {self.output_file} ...')
            np.savez(self.output_file, **loaded_data)
            print('... done!')

        return loaded_data

    def _load_all_sparse(self, r_cut: float) -> List[jraph.GraphsTuple]:
        """Load from ASE file and convert to list jraph.GraphTuples.

        Args:
            r_cut (): Cutoff radius for calculation of neighbors.

        Returns: List of `jraph.GraphTuples`.

        """

        print(f"Read data from {self.input_file} ...")
        loaded_data = []

        if self.load_stress:
            raise NotImplementedError(f'Loading stress for `neighbors_format={self.neighbors_format}` is not '
                                      f'supported yet.')

        for a in tqdm(iread(self.input_file)):
            graph = ASE_to_jraph(a, r_cut=r_cut)
            loaded_data.append(graph)
        print("... done!")

        return loaded_data

    def load_all(self, r_cut: float = None):
        if self.neighbors_format.lower() == 'dense':
            return self._load_all_dense()
        elif self.neighbors_format.lower() == 'sparse':
            if r_cut is None:
                raise ValueError(
                    f'When using `self.neighbors_format=sparse`, please pass `r_cut` to `load_all()` method.')
            return self._load_all_sparse(r_cut=r_cut)
        else:
            raise ValueError(f'{self.neighbors_format} is invalid neighbors_format. Try `dense` or `sparse.`')


def ASE_to_jraph(
    mol: Atoms,
    r_cut: float,
    self_interaction: bool = False,
) -> jraph.GraphsTuple:
    """
    Convert an ASE Atoms object to a jraph.GraphTuple object.

    Parameters:
    - mol (Atoms): ASE Atoms object.
    - r_cut (float): Cutoff radius for neighbor interactions.
    - self_interaction (bool): Include self-interaction in neighbor list.

    Returns:
    - jraph.GraphsTuple: Jraph graph representation of the Atoms object.
    """
    energy = mol.get_potential_energy()
    atomic_numbers = mol.get_atomic_numbers()
    positions = mol.get_positions()
    forces = mol.get_forces()

    if mol.get_pbc().any():
        # TODO: Add shift vector to jraph.GraphsTuple for PBCs. Should go to edge_features.
        raise NotImplementedError(f'PBCs for `pbc={mol.get_pbc().any()}` is not '
                                  f'supported yet.')

    i, j = neighbor_list('ij', mol, r_cut, self_interaction=self_interaction)

    node_features = {
            "atomic_positions": jnp.array(positions),
            "atomic_numbers": jnp.array(atomic_numbers),
            "forces": jnp.array(forces),
            }

    senders = jnp.array(i)
    receivers = jnp.array(j)

    n_node = jnp.array([len(node_features)])
    n_edge = jnp.array([len(i)])

    global_context = {"energy": jnp.array([energy])},

    graph = jraph.GraphsTuple(
        nodes=node_features,
        edges=None,
        senders=senders,
        receivers=receivers,
        n_node=n_node,
        n_edge=n_edge,
        globals=global_context
    )

    return graph
