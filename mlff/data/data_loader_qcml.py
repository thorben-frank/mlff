import h5py
import jax.numpy as jnp
from dataclasses import dataclass
import jraph
from tqdm import tqdm
import numpy as np
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
class QCMLLoaderSparse:
    # Loader for quantum chemistry machine learning data.
    input_file: str
    min_distance_filter: float = 0.
    max_force_filter: float = 1.e6

    def cardinality(self):
        data = h5py.File(self.input_file)
        n = 0
        for i in data:
            n += 1
        return n

    def load(self, cutoff: float, pick_idx: np.ndarray = None):
        if pick_idx is None:
            def keep(idx: int):
                return True
        else:
            def keep(idx: int):
                return idx in pick_idx

        data = h5py.File(self.input_file)

        max_num_of_nodes = 0
        max_num_of_edges = 0
        logging.mlff(
            f"Load data from {self.input_file} and calculate neighbors within cutoff={cutoff} Ang ..."
        )
        loaded_data = []
        i = 0
        for k in tqdm(data):
            if keep(i):
                positions = np.array(data[k]['positions'])
                atomic_numbers = np.array(data[k]['atomic_numbers'])
                forces = data[k]['forces']
                energy = data[k]['energy']
                charge = data[k]['charge']
                multiplicity = data[k]['multiplicity']
                if len(atomic_numbers) > 1:
                    senders, receivers, minimal_distance = compute_senders_and_receivers_np(
                        positions,
                        cutoff=cutoff
                    )

                    if (
                            minimal_distance < self.min_distance_filter or
                            np.abs(forces).max() > self.max_force_filter
                    ):
                        g = None
                    else:
                        g = jraph.GraphsTuple(
                            n_node=np.array([len(atomic_numbers)]),
                            n_edge=np.array([len(receivers)]),
                            globals=dict(
                                energy=np.array(energy).reshape(-1),
                                total_charge=np.array(charge).reshape(-1),
                                num_unpaired_electrons=np.array(multiplicity).reshape(-1) - 1
                            ),
                            nodes=dict(
                                atomic_numbers=atomic_numbers.reshape(-1).astype(np.int16),
                                positions=positions,
                                forces=np.array(forces)
                            ),
                            edges=dict(cell=None, cell_offsets=None),
                            receivers=np.array(senders),  # opposite convention in mlff
                            senders=np.array(receivers)
                        )
                else:
                    g = None

                loaded_data += [g]
                if g is not None:
                    num_nodes = len(g.nodes['atomic_numbers'])
                    num_edges = len(g.receivers)
                    max_num_of_nodes = max_num_of_nodes if num_nodes <= max_num_of_nodes else num_nodes
                    max_num_of_edges = max_num_of_edges if num_edges <= max_num_of_edges else num_edges
            else:
                pass
            i += 1

        if pick_idx is not None:
            if max(pick_idx) >= i:
                raise RuntimeError(
                    f'`max(pick_idx) = {max(pick_idx)} >= cardinality = {i} of the dataset`.'
                )

        logging.mlff("... done!")

        return loaded_data, {'max_num_of_nodes': max_num_of_nodes, 'max_num_of_edges': max_num_of_edges}
