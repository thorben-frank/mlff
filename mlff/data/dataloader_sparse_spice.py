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


def compute_senders_and_receivers(
        positions, cutoff: float
):
    num_atoms = positions.shape[0]
    displacements = positions[None, :, :] - positions[:, None, :]
    distances = jnp.linalg.norm(displacements, axis=-1)
    mask = ~jnp.eye(num_atoms, dtype=jnp.bool_) # get rid of self interactions
    keep_edges = jnp.where((distances < cutoff) & mask)
    senders = keep_edges[0].astype(jnp.int32)
    receivers = keep_edges[1].astype(jnp.int32)
    return senders, receivers


def compute_senders_and_receivers_np(
    positions, cutoff: float
):
    """Computes an edge list from atom positions and a fixed cutoff radius."""
    num_atoms = positions.shape[0]
    displacements = positions[None, :, :] - positions[:, None, :]
    distances = np.linalg.norm(displacements, axis=-1)
    mask = ~np.eye(num_atoms, dtype=np.bool_) # get rid of self interactions
    keep_edges = np.where((distances < cutoff) & mask)
    senders = keep_edges[0].astype(np.int32)
    receivers = keep_edges[1].astype(np.int32)
    return senders, receivers


@dataclass
class SpiceDataLoaderSparse:
    input_file: str

    def load_all(self, cutoff: float):
        data = h5py.File(self.input_file)

        max_num_of_nodes = 0
        max_num_of_edges = 0
        logging.mlff(
            f"Load data from {self.input_file} and calculate neighbors within cutoff={cutoff} Ang ..."
        )
        loaded_data = []
        for i in tqdm(data):
            conformations = data[i]['conformations']
            atomic_numbers = data[i]['atomic_numbers']
            forces = data[i]['dft_total_gradient']
            energy = data[i]['dft_total_energy']
            for n in range(len(conformations)):
                senders, receivers = compute_senders_and_receivers_np(conformations[n], cutoff=cutoff)

                g = jraph.GraphsTuple(
                    n_node=jnp.array([len(atomic_numbers)]),
                    n_edge=jnp.array([len(receivers)]),
                    globals=dict(energy=jnp.array(energy[n]).reshape(-1)),
                    nodes=dict(
                        atomic_numbers=jnp.array(atomic_numbers).astype(jnp.int16),
                        positions=jnp.array(conformations[n]),
                        forces=jnp.array(forces[n])
                    ),
                    edges=None,
                    receivers=jnp.array(senders),  # opposite convention in mlff
                    senders=jnp.array(receivers)
                )
                loaded_data += [g]
                num_nodes = len(g.nodes['atomic_numbers'])
                num_edges = len(g.receivers)
                max_num_of_nodes = max_num_of_nodes if num_nodes <= max_num_of_nodes else num_nodes
                max_num_of_edges = max_num_of_edges if num_edges <= max_num_of_edges else num_edges

        logging.mlff("... done!")

        return loaded_data, {'max_num_of_nodes': max_num_of_nodes, 'max_num_of_edges': max_num_of_edges}
