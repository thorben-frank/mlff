from ase.io import iread
from ase.neighborlist import neighbor_list
from ase import Atoms
from dataclasses import dataclass
from typing import List
from tqdm import tqdm
import numpy as np
import jraph

@dataclass
class AseDataLoader:
    input_file: str
    output_file: str = None
    load_stress: bool = False
    load_energy_and_forces: bool = True

    def load_all(self) -> List:

        def ASE_to_jraph(
            mol: Atoms,
            r_cut: float = 5.0,
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

            i, j = neighbor_list('ij', mol, r_cut, self_interaction=self_interaction)

            node_features = {
                    "atomic_positions": np.array(positions),
                    "atomic_numbers": np.array(atomic_numbers),
                    "forces": np.array(forces),
                    }

            senders = np.array(i)
            receivers = np.array(j)

            n_node = np.array([len(node_features)])
            n_edge = np.array([len(i)])

            global_context = {"energy": np.array([energy])},

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

        print(f"Read data from {self.input_file} ...")
        loaded_data = []
        for a in tqdm(iread(self.input_file)):
            graph = ASE_to_jraph(a)
            loaded_data.append(graph)
        print("... done!")

        return loaded_data
