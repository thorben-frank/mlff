from ase.calculators.calculator import PropertyNotImplementedError
from ase.io import read, iread
from ase.neighborlist import neighbor_list
from ase import Atoms
from dataclasses import dataclass
from typing import Optional
from tqdm import tqdm
import numpy as np
import jraph
import os

import logging

from functools import partial, partialmethod

logging.MLFF = 35
logging.addLevelName(logging.MLFF, 'MLFF')
logging.Logger.trace = partialmethod(logging.Logger.log, logging.MLFF)
logging.mlff = partial(logging.log, logging.MLFF)


def compute_senders_and_receivers_np(
    positions, cutoff: float
):
    """Computes an edge list from atom positions and a fixed cutoff radius."""
    num_atoms = positions.shape[0]
    displacements = positions[None, :, :] - positions[:, None, :]
    distances = np.linalg.norm(displacements, axis=-1)
    distances_min = distances[distances > 0].min()
    mask = ~np.eye(num_atoms, dtype=np.bool_)  # get rid of self interactions
    keep_edges = np.where((distances < cutoff) & mask)
    senders = keep_edges[0].astype(np.int32)
    receivers = keep_edges[1].astype(np.int32)
    return senders, receivers, distances_min


@dataclass
class AseDataLoaderSparse:
    input_file: Optional[str] = None,
    input_folder: Optional[str] = None,
    min_distance_filter: float = 0.
    max_force_filter: float = 1.e6


    def cardinality(self):
        print(self.input_folder)
        if self.input_folder[0] is not None:
            file_list = [f for f in os.listdir(self.input_folder) if os.path.isfile(os.path.join(self.input_folder, f))]
            print(file_list)
            total_atoms = 0
            for file in file_list:
                atoms = read(os.path.join(self.input_folder, file), index=":", format='extxyz')
                total_atoms += len(atoms)
            return total_atoms
        elif self.input_file:
            atoms = read(self.input_file, index=":", format='extxyz')
            return len(atoms)

    def load(self, cutoff: float, pick_idx: np.ndarray = None):
        if pick_idx is None:
            def keep(idx: int):
                return True
        else:
            def keep(idx: int):
                return idx in pick_idx

        if self.input_folder[0] is not None:
        #if self.input_folder is not None:
            file_list = [os.path.join(self.input_folder, f) for f in os.listdir(self.input_folder) if os.path.isfile(os.path.join(self.input_folder, f))]
        elif self.input_file:
            file_list = [self.input_file]
        
        loaded_data = []
        max_num_of_nodes = 0
        max_num_of_edges = 0
        max_num_of_pairs = 0
        logging.mlff(f"Loading data from file_list: {file_list} ...")

        i = 0
        for file_path in file_list:
            logging.mlff(
                f"Load data from {file_path} and calculate neighbors within cutoff={cutoff} Ang ..."
            )
            for a in tqdm(iread(file_path, format='extxyz'), mininterval = 60, maxinterval=600):
                if keep(i):
                    graph = ASE_to_jraph(
                        a,
                        min_distance_filter=self.min_distance_filter,
                        max_force_filter=self.max_force_filter,
                        cutoff=cutoff
                    )

                    loaded_data.append(graph)

                    if graph is not None:
                        num_nodes = len(graph.nodes['atomic_numbers'])
                        num_edges = len(graph.receivers)
                        num_pairs = num_nodes * (num_nodes - 1)
                        max_num_of_nodes = max_num_of_nodes if num_nodes <= max_num_of_nodes else num_nodes
                        max_num_of_edges = max_num_of_edges if num_edges <= max_num_of_edges else num_edges
                        max_num_of_pairs = max_num_of_pairs if num_pairs <= max_num_of_pairs else num_pairs
                else:
                    pass
                i += 1

        if pick_idx is not None:
            if max(pick_idx) >= i:
                raise RuntimeError(
                    f'`max(pick_idx) = {max(pick_idx)} >= cardinality = {i} of the dataset`.'
                )
        logging.mlff("... done!")

        return loaded_data, {'max_num_of_nodes': max_num_of_nodes, 'max_num_of_edges': max_num_of_edges, 'max_num_of_pairs': max_num_of_pairs}

def ASE_to_jraph(
    mol: Atoms,
    cutoff: float,
    min_distance_filter: float,
    max_force_filter: float,
    self_interaction: bool = False,
):
    """Convert an ASE Atoms object to a jraph.GraphTuple object.

    Args:
        mol (Atoms): ASE Atoms object.
        cutoff (float): Cutoff radius for neighbor interactions.
        min_distance_filter (float):
        max_force_filter (float):
        self_interaction (bool): Include self-interaction in neighbor list.

    Returns:
        jraph.GraphsTuple: Jraph graph representation of the Atoms object if filter != True else None.
    """

    atomic_numbers = mol.get_atomic_numbers()
    n_atoms = len(atomic_numbers)
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
        #TODO: Read Hirshfeld ratios only when they are needed, 
        #Now they are set to 0 if not present 
        #hirsh_bool is used in observable_funnction_sparse.py before passing values to loss function
        try:
            hirshfeld_ratios = mol.arrays['hirsh_ratios']
        except:
            hirshfeld_ratios = [0.] * n_atoms
        try:
            dipole = mol.get_dipole_moment()
        except:
            dipole = None
        try:
            total_charge = mol.info['charge']
        except:
            total_charge = 0.
        try:
            multiplicity = mol.info['multiplicity']
        except:
            multiplicity = 1

    else:
        energy = None
        forces = None
        stress = None
        hirshfeld_ratios = None
        dipole = None
        total_charge = None
        multiplicity = None

    if mol.get_pbc().any():
        i, j, S = neighbor_list('ijS', mol, cutoff, self_interaction=self_interaction)
        cell = np.array(mol.get_cell())
        edge_features = {
            "cell": np.repeat(np.array(cell)[None], repeats=len(S), axis=0),
            "cell_offset": np.array(S)
        }
        senders = np.array(j)
        receivers = np.array(i)
    else:
        i, j = neighbor_list('ij', mol, cutoff, self_interaction=self_interaction)
        edge_features = {
            "cell": None,
            "cell_offset": None
        }

        senders = np.array(j)
        receivers = np.array(i)

        # if len(atomic_numbers) == 1:
        #     return None

        # senders, receivers, minimal_distance = compute_senders_and_receivers_np(
        #     positions,
        #     cutoff=cutoff
        # )

        # if (
        #         minimal_distance < min_distance_filter or
        #         np.abs(forces).max() > max_force_filter
        # ):
        #     return None

    node_features = {
            "positions": np.array(positions),
            "atomic_numbers": np.array(atomic_numbers),
            "forces": np.array(forces) if forces is not None else None,
            "hirshfeld_ratios": np.array(hirshfeld_ratios) if hirshfeld_ratios is not None else None
            }
    
    idx_i_lr, idx_j_lr = neighbor_list('ij', mol, 100, self_interaction=self_interaction)
    idx_i_lr = np.array(idx_i_lr)
    idx_j_lr = np.array(idx_j_lr)

    # n_node = np.array([mol.get_global_number_of_atoms()])
    n_node = np.array([n_atoms])

    n_edge = np.array([len(senders)])
    n_pairs = np.array([len(idx_i_lr)])


    global_context = {
        "energy": np.array([energy]) if energy is not None else None,
        "stress": np.array(stress) if stress is not None else None,
        "dipole_vec": np.array(dipole.reshape(-1,3)) if dipole is not None else None,
        "total_charge": np.array(total_charge, dtype=np.int16).reshape(-1) if total_charge is not None else None,
        "hirsh_bool": np.array([0]) if hirshfeld_ratios[0]==0. else np.array([1]),
        "num_unpaired_electrons": np.array([multiplicity]) - 1 if multiplicity is not None else None,
    }

    return jraph.GraphsTuple(
                nodes=node_features,
                edges=edge_features,
                senders=senders,
                receivers=receivers,
                n_node=n_node,
                n_edge=n_edge,
                globals=global_context,
                n_pairs = n_pairs,
                idx_i_lr = idx_i_lr,
                idx_j_lr = idx_j_lr
    )
