from ase.calculators.calculator import PropertyNotImplementedError
from ase.io import read, iread
from ase.io.formats import ioformats
from ase.neighborlist import neighbor_list
from ase import Atoms
from dataclasses import dataclass
from typing import Optional
from tqdm import tqdm
import numpy as np
import jraph
import os
from pathlib import Path

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
    input_file: Optional[str] = None
    input_folder: Optional[str] = None

    def make_file_list(self):
        if self.input_folder is not None:
            if self.input_file is not None:
                raise ValueError(
                    f"Only input_folder or input_file can be specified. "
                    f"Received {self.input_folder=} and {self.input_file=}."
                )
            input_folder = Path(self.input_folder).expanduser().resolve()
            file_list = []
            for entry in os.scandir(input_folder):
                if Path(entry.path).suffix[1:] in set(ioformats.keys()):
                    file_list.append(Path(entry.path).expanduser().resolve())
        else:
            if self.input_file is None:
                raise ValueError(
                    f"Either input_folder or input_file must be set. "
                    f"Received {self.input_folder=} and {self.input_file=}."
                )
            input_file = Path(self.input_file).expanduser().resolve()
            file_list = [input_file]

        return file_list

    def cardinality(self):
        file_list = self.make_file_list()
        total_atoms = 0
        for file in file_list:
            atoms = ['' for _ in iread(file)]
            total_atoms += len(atoms)
        return total_atoms

    def load(
            self,
            cutoff: float,
            calculate_neighbors_lr: bool = False,
            cutoff_lr: Optional[float] = None,
            pick_idx: np.ndarray = None
    ):
        if pick_idx is None:
            def keep(idx: int):
                return True
        else:
            def keep(idx: int):
                return idx in pick_idx

        loaded_data = []
        max_num_of_nodes = 0
        max_num_of_edges = 0
        max_num_of_pairs = 0

        file_list = self.make_file_list()

        logging.mlff(f"Loading data from file_list: {file_list} ...")

        i = 0
        for file_path in file_list:
            logging.mlff(
                f"Load data from {file_path}."
            )
            logging.mlff(
                f"Calculate short range neighbors within cutoff={cutoff} Ang."
            )
            if calculate_neighbors_lr:
                logging.mlff(
                    f"Calculate long-range neighbors within cutoff_lr={cutoff_lr} Ang."
                )
            for a in tqdm(iread(file_path), mininterval=60, maxinterval=600):
                if keep(i):
                    graph = ASE_to_jraph(
                        a,
                        cutoff=cutoff,
                        calculate_neighbors_lr=calculate_neighbors_lr,
                        cutoff_lr=cutoff_lr
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
        self_interaction: bool = False,
        calculate_neighbors_lr: bool = True,
        cutoff_lr: float = Optional[None]
):
    """Convert an ASE Atoms object to a jraph.GraphTuple object.

    Args:
        mol (Atoms): ASE Atoms object.
        cutoff (float): Cutoff radius for neighbor interactions.
        self_interaction (bool): Include self-interaction in neighbor list.
        calculate_neighbors_lr (bool): Calculate long-range neighborhood.
        cutoff_lr (float): Cutoff for the long-range neighborhood.

    Returns:
        jraph.GraphsTuple: Jraph graph representation of the Atoms object if filter != True else None.
    """

    atomic_numbers = np.array(mol.get_atomic_numbers(), dtype=np.int64)
    positions = np.array(mol.get_positions())

    num_atoms = len(atomic_numbers)

    if mol.get_calculator() is not None:
        try:
            energy = np.array(mol.get_potential_energy()).reshape(-1)
        except PropertyNotImplementedError:
            energy = None
        try:
            forces = np.array(mol.get_forces())
        except PropertyNotImplementedError:
            forces = None
        try:
            stress = np.array(mol.get_stress())
        except PropertyNotImplementedError:
            stress = None
        try:
            dipole = np.array(mol.get_dipole_moment())
        except PropertyNotImplementedError:
            dipole = None
    else:
        energy = None
        forces = None
        stress = None
        dipole = None

    # Forces are NaN when not present.
    if forces is None:
        forces = np.empty((num_atoms, 3))
        forces[:] = np.nan

    # Energy is NaN when not present.
    if energy is None:
        # Energy from ASE is only a scalar.
        energy = np.nan

    # Dipoles are NaN when not present.
    if dipole is None:
        dipole = np.empty((3, ))
        dipole[:] = np.nan

    # Stress is NaN when not present.
    if stress is None:
        stress = np.empty((6, ))
        stress[:] = np.nan

    # Read additional properties from .info and .arrays in Atoms object.
    total_charge = mol.info.get('charge')
    multiplicity = mol.info.get('multiplicity')
    hirshfeld_ratios = mol.arrays.get('hirsh_ratios')

    # Total charges are assumed to be zero when not specified.
    if total_charge is None:
        total_charge = np.array(0, np.int16).reshape(-1)
    else:
        total_charge = np.array(total_charge, dtype=np.int16).reshape(-1)

    # Multiplicity is assumed to be one when not specified.
    if multiplicity is None:
        multiplicity = np.array(1, dtype=np.int16).reshape(-1)
    else:
        multiplicity = np.array(multiplicity, dtype=np.int16).reshape(-1)

    # Hirshfeld ratios are set to NaN when not present.
    if hirshfeld_ratios is None:
        hirshfeld_ratios = np.empty((num_atoms, ))
        hirshfeld_ratios[:] = np.nan
    else:
        hirshfeld_ratios = np.array(hirshfeld_ratios).reshape(num_atoms, )

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
        edge_features = dict()  # No edge features.

        senders = np.array(j)
        receivers = np.array(i)

    if calculate_neighbors_lr:
        if mol.get_pbc().any():
            raise NotImplementedError(
                'Long-range neighborhoods can only be calculated for non-PBC at the moment.'
            )
        if cutoff_lr is None:
            raise ValueError(
                f'cutoff_lr must be specified for {calculate_neighbors_lr=}. Received {cutoff_lr=}.'
            )
        idx_i_lr, idx_j_lr = neighbor_list('ij', mol, cutoff_lr, self_interaction=self_interaction)
        idx_i_lr = np.array(idx_i_lr)
        idx_j_lr = np.array(idx_j_lr)
    else:
        # No long-range indices are calculated.
        idx_i_lr = np.array([])
        idx_j_lr = np.array([])

    n_node = np.array([num_atoms])

    n_edge = np.array([len(senders)])

    # Evaluates to zero if long-range edges are not calculated. When batching the graphs, this ensures that
    # n_pairs can not lead to overflow when no long-range indices are present.
    n_pairs = np.array([len(idx_i_lr)])

    # Add axis one for vector quantities such that batching yields e.g. (num_graphs, 6) for stress and (num_graphs, 3)
    # for dipole_vec. Otherwise batching via jraph yields (num_graphs*6, ) and (num_graphs*3, ). This is ultimately
    # convention that has to be taken into account when calculating the loss. Scalars like energy are represented
    # as (num_graphs, ).
    global_context = {
        "energy": energy.reshape(-1),
        "stress": stress.reshape(1, 6),
        "dipole_vec": dipole.reshape(1, 3),
        "total_charge": total_charge.reshape(-1),
        "num_unpaired_electrons": multiplicity.reshape(-1) - 1,
    }

    # Edges follow a similar convention where e.g. for positions and forces one has (num_nodes, 3) and for scalars
    # like hirshfeld volumes (num_nodes, ).
    node_features = {
            "positions": positions.reshape(num_atoms, 3),
            "atomic_numbers": atomic_numbers.reshape(num_atoms),
            "forces": forces.reshape(num_atoms, 3),
            "hirshfeld_ratios": hirshfeld_ratios.reshape(num_atoms)
            }

    return jraph.GraphsTuple(
                nodes=node_features,
                edges=edge_features,
                senders=senders,
                receivers=receivers,
                n_node=n_node,
                n_edge=n_edge,
                globals=global_context,
                n_pairs=n_pairs,
                idx_i_lr=idx_i_lr,
                idx_j_lr=idx_j_lr
    )
