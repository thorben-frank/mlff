from dataclasses import dataclass
import jraph
from tqdm import tqdm
import numpy as np
import logging
import os
import pathlib

from functools import partial, partialmethod

try:
    import tensorflow as tf
except ModuleNotFoundError:
    logging.warning(
        "For using TFDSDataLoader please install tensorflow."
    )
    # raise RuntimeWarning(
    #     "For using TFRecordDataLoaderSparse please install tensorflow."
    # )

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
class TFRecordDataLoaderSparse:
    input_file: str
    min_distance_filter: float = 0.,
    max_force_filter: float = 1.e6

    def cardinality(self):
        n = 0
        for entry in os.scandir(self.input_file):
            entry_path = pathlib.Path(entry.path).resolve()
            if entry_path.suffix[:9] == '.tfrecord':
                dataset = tf.data.TFRecordDataset(entry_path)
                n += sum(1 for _ in dataset)
            else:
                pass

        return n

    def load(
            self,
            cutoff: float,
            pick_idx: np.ndarray = None,
    ):
        if pick_idx is None:
            def keep(idx: int):
                return True
        else:
            def keep(idx: int):
                return idx in pick_idx

        max_num_of_nodes = 0
        max_num_of_edges = 0
        logging.mlff(
            f"Load data from {self.input_file} and calculate neighbors within cutoff={cutoff} ..."
        )
        loaded_data = []
        i = 0
        num_skip = 0
        for entry in os.scandir(self.input_file):
            entry_path = pathlib.Path(entry.path).resolve()
            if entry_path.suffix[:9] == '.tfrecord':
                dataset = tf.data.TFRecordDataset(entry_path)
                for record in tqdm(dataset):
                    if keep(i):
                        g, skip = entry_to_jraph(
                            record,
                            cutoff=cutoff,
                            min_distance_filter=self.min_distance_filter,
                            max_force_filter=self.max_force_filter
                        )

                        loaded_data += [g]

                        if skip:
                            num_skip += 1
                        else:
                            num_nodes = len(g.nodes['atomic_numbers'])
                            num_edges = len(g.receivers)
                            max_num_of_nodes = max_num_of_nodes if num_nodes <= max_num_of_nodes else num_nodes
                            max_num_of_edges = max_num_of_edges if num_edges <= max_num_of_edges else num_edges
                    else:
                        pass
                    i += 1
            else:
                print(f'Skipping {entry_path} since it is not a .tfrecord file.')

        if pick_idx is not None:
            if max(pick_idx) >= i:
                raise RuntimeError(
                    f'`max(pick_idx) = {max(pick_idx)} >= cardinality = {i} of the dataset`.'
                )

        logging.mlff("... done!")

        return loaded_data, {'max_num_of_nodes': max_num_of_nodes, 'max_num_of_edges': max_num_of_edges}


def parse_tf_feature(feature):
    if feature.HasField('bytes_list'):
        return feature.bytes_list.value
    elif feature.HasField('float_list'):
        return feature.float_list.value
    elif feature.HasField('int64_list'):
        return feature.int64_list.value
    else:
        return None


def entry_to_jraph(
        entry,
        cutoff: float,
        min_distance_filter: float,
        max_force_filter: float
):
    example = tf.train.Example()
    example.ParseFromString(entry.numpy())
    atomic_numbers = np.array(parse_tf_feature(example.features.feature['atomic_numbers'])).reshape(-1)
    num_unpaired_electrons = np.array(parse_tf_feature(example.features.feature['multiplicity'])).reshape(-1) - 1
    energy = np.array(parse_tf_feature(example.features.feature['formation_energy'])).reshape(-1)
    forces = np.array(parse_tf_feature(example.features.feature['forces'])).reshape(len(atomic_numbers), 3)
    total_charge = np.array(parse_tf_feature(example.features.feature['charge'])).reshape(-1)
    positions = np.array(parse_tf_feature(example.features.feature['positions'])).reshape(len(atomic_numbers), 3)

    if len(atomic_numbers) > 1:
        senders, receivers, minimal_distance = compute_senders_and_receivers_np(positions, cutoff=cutoff)
    else:
        return None, True

    if (
            minimal_distance < min_distance_filter or
            np.abs(forces).max() > max_force_filter
    ):
        return None, True

    g = jraph.GraphsTuple(
        n_node=np.array([len(atomic_numbers)]),
        n_edge=np.array([len(receivers)]),
        globals=dict(
            energy=energy,
            total_charge=total_charge,
            num_unpaired_electrons=num_unpaired_electrons,
        ),
        nodes=dict(
            atomic_numbers=atomic_numbers.astype(np.int16),
            positions=positions,
            forces=forces
        ),
        edges=dict(cell=None, cell_offsets=None),
        receivers=senders,  # opposite convention in mlff
        senders=receivers
    )
    return g, False
