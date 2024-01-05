from ase.calculators.calculator import PropertyNotImplementedError
from ase.io import iread
from ase.neighborlist import neighbor_list
from ase import Atoms
from dataclasses import dataclass
from typing import List
from tqdm import tqdm
import jax.numpy as jnp
import numpy as np
import jraph


@dataclass
class AseDataLoaderSparse:
    input_file: str

    def load_all(self, cutoff: float) -> List:
        print(f"Read data from {self.input_file} ...")
        loaded_data = []
        for a in tqdm(iread(self.input_file)):
            graph = ASE_to_jraph(a, cutoff=cutoff)
            loaded_data.append(graph)
        print("... done!")

        return loaded_data


def ASE_to_jraph(
    mol: Atoms,
    cutoff: float,
    self_interaction: bool = False,
) -> jraph.GraphsTuple:
    """Convert an ASE Atoms object to a jraph.GraphTuple object.

    Args:
        mol (Atoms): ASE Atoms object.
        cutoff (float): Cutoff radius for neighbor interactions.
        self_interaction (bool): Include self-interaction in neighbor list.

    Returns:
        jraph.GraphsTuple: Jraph graph representation of the Atoms object.
    """

    atomic_numbers = mol.get_atomic_numbers()
    positions = mol.get_positions()
    if mol.get_calculator() is not None:
        try:
            energy = mol.get_potential_energy()
        except PropertyNotImplementedError:
            energy = None
        try:
            forces = mol.get_forces()
        except PropertyNotImplementedError:
            forces = None
        try:
            stress = mol.get_stress()
        except PropertyNotImplementedError:
            stress = None
    else:
        energy = None
        forces = None
        stress = None

    if mol.get_pbc().any():
        i, j, S = neighbor_list('ijS', mol, cutoff, self_interaction=self_interaction)
        cell = np.array(mol.get_cell())
        edge_features = {
            "cell": jnp.array(cell),
            "cell_offset": jnp.array(S)
        }
    else:
        i, j = neighbor_list('ij', mol, cutoff, self_interaction=self_interaction)
        edge_features = {
            "cell": None,
            "cell_offset": None
        }

    node_features = {
            "positions": jnp.array(positions),
            "atomic_numbers": jnp.array(atomic_numbers),
            "forces": jnp.array(forces) if forces is not None else None,
            }

    senders = jnp.array(j)
    receivers = jnp.array(i)

    n_node = jnp.array([mol.get_global_number_of_atoms()])
    n_edge = jnp.array([len(i)])

    global_context = {
        "energy": jnp.array([energy]) if energy is not None else None,
        "stress": jnp.array(stress) if stress is not None else None
    }

    return jraph.GraphsTuple(
                nodes=node_features,
                edges=edge_features,
                senders=senders,
                receivers=receivers,
                n_node=n_node,
                n_edge=n_edge,
                globals=global_context
    )
