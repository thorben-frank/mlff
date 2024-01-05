from ase.neighborlist import neighbor_list
from ase import Atoms
from dataclasses import dataclass
from typing import List
import jax.numpy as jnp
import jraph
import numpy as np


@dataclass
class NpzDataLoaderSparse:
    input_file: str

    def load_all(self, cutoff: float) -> List:
        print(f"Read data from {self.input_file} ...")
        np_data = np.load(self.input_file)

        keys = list(np_data.keys())
        data_iterator = zip(*(np_data[k] for k in keys))

        loaded_data = []
        for i, tup in enumerate(data_iterator):
            entry = {k: v for (k, v) in zip(keys, tup)}
            graph = entry_to_jraph(entry, cutoff=cutoff)
            loaded_data.append(graph)

        print("... done!")

        return loaded_data


def entry_to_jraph(
    entry,
    cutoff: float,
    self_interaction: bool = False,
) -> jraph.GraphsTuple:
    """Convert an entry to a jraph.GraphTuple object.

    Args:
        entry (Dict): Entry.
        cutoff (float): Cutoff radius for neighbor interactions.
        self_interaction (bool): Include self-interaction in neighbor list.

    Returns:
        jraph.GraphsTuple: Jraph graph representation of the Atoms object.
    """

    atomic_numbers = entry['atomic_numbers']
    positions = entry['positions']
    forces = entry.get('forces')
    energy = entry.get('energy')
    stress = entry.get('stress')
    cell = entry.get('cell')
    pbc = entry.get('pbc')
    mol = Atoms(positions=positions, numbers=atomic_numbers, cell=cell, pbc=pbc)

    if mol.get_pbc().any():
        i, j, S = neighbor_list('ijS', mol, cutoff, self_interaction=self_interaction)
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
        "atomic_numbers": jnp.array(atomic_numbers, dtype=jnp.int32),
        "forces": jnp.array(forces),
            }

    senders = jnp.array(j)
    receivers = jnp.array(i)

    n_node = jnp.array([mol.get_global_number_of_atoms()])
    n_edge = jnp.array([len(i)])

    global_context = {
        "energy": jnp.array(energy) if energy is not None else None,
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
