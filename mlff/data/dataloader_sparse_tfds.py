from dataclasses import dataclass
import logging
import jraph

from functools import partial, partialmethod
from typing import Optional

try:
    import tensorflow as tf
except ModuleNotFoundError:
    logging.warning(
        "For using TFDSDataLoader please install tensorflow."
    )
    # raise RuntimeWarning(
    #     "For using TFDSDataLoader please install tensorflow."
    # )
try:
    import tensorflow_datasets as tfds
except ModuleNotFoundError:
    logging.warning(
        "For using TFDSDataLoader please install tensorflow_datasets."
    )
    # raise RuntimeWarning(
    #     "For using TFDSDataLoader please install tensorflow_datasets."
    # )


logging.MLFF = 35
logging.addLevelName(logging.MLFF, 'MLFF')
logging.Logger.trace = partialmethod(logging.Logger.log, logging.MLFF)
logging.mlff = partial(logging.log, logging.MLFF)


def compute_edges_tf(
    positions, cutoff: float
):
    num_atoms = tf.shape(positions)[0]
    displacements = positions[None, :, :] - positions[:, None, :]
    distances = tf.norm(displacements, axis=-1)
    mask = ~tf.eye(num_atoms, dtype=tf.bool)  # Get rid of self-connections.
    keep_edges = tf.where((distances < cutoff) & mask)
    senders = tf.cast(keep_edges[:, 0], dtype=tf.int32)
    receivers = tf.cast(keep_edges[:, 1], dtype=tf.int32)
    return senders, receivers, (tf.math.reduce_min(distances), tf.math.reduce_max(distances))


# TODO: include n_pairs
def create_graph_tuple_tf(
        element, cutoff: float
) -> jraph.GraphsTuple:
    """Takes a data element and wraps relevant components in a GraphsTuple."""
    nodes_dict = dict()
    globals_dict = dict()

    atomic_numbers = element['atomic_numbers']
    positions = element['positions']

    nodes_dict['positions'] = positions
    nodes_dict['atomic_numbers'] = atomic_numbers

    properties = element.keys()

    if 'energy' in properties:
        globals_dict['energy'] = tf.reshape(element['formation_energy'], (1,))
    if 'forces' in properties:
        nodes_dict['forces'] = element['forces']
    if 'multiplicity' in properties:
        globals_dict['num_unpaired_electrons'] = tf.reshape(element['multiplicity'], (1,)) - 1
    if 'charge' in properties:
        globals_dict['total_charge'] = tf.reshape(element['charge'], (1,))

    senders, receivers, (min_distance, max_distance) = compute_edges_tf(
        positions, cutoff
    )

    num_nodes = tf.shape(atomic_numbers)[0]
    num_edges = tf.shape(senders)[0]
    return jraph.GraphsTuple(
        n_node=tf.reshape(num_nodes, (1,)),
        n_edge=tf.reshape(num_edges, (1,)),
        senders=senders,
        receivers=receivers,
        nodes=nodes_dict,
        globals=globals_dict,
        edges=dict(),  # Don't set to None, since otherwise tf.data.Dataset.to_numpy_generator() does not work due to
        # call to None.numpy().
    )


# TODO: Think about renaming QCMLDataLoader
@dataclass
class TFDSDataLoaderSparse:
    input_file: str
    split: str = 'train'
    max_force_filter: float = 1.e6

    def cardinality(self):
        builder = tfds.builder_from_directory(
            self.input_file
        )

        # Metadata are available as usual
        num_data = builder.info.splits[self.split].num_examples
        del builder

        return num_data

    # TODO: Control everything via self.split, e.g. split = ['train[:50]', 'valid[:10]'] which can be constructed
    #  using string formatting. Allows full control compatible with tf.data.Dataset API and num_train, num_valid are
    #  not required anymore.
    def load(
            self,
            cutoff: float,
            num_train: int,
            num_valid: int,
            num_test: Optional[int] = None,
            return_test: bool = False
    ):
        builder = tfds.builder_from_directory(
            self.input_file
        )

        test_string = f'{self.split}[{num_train}:{-num_valid}]' if num_test is None else f'{self.split}[{-(num_valid+num_test)}:{-num_valid}]'

        test_ds = None
        if return_test:
            # Split into train, valid and test.
            train_ds, test_ds, valid_ds = builder.as_dataset(
                split=[
                    f'{self.split}[:{num_train}]',
                    test_string,
                    f'{self.split}[{-num_valid}:]'
                ]
            )
        else:
            # Split into train and valid.
            train_ds, valid_ds = builder.as_dataset(
                split=[
                    f'{self.split}[:{num_train}]',
                    f'{self.split}[{-num_valid}:]'
                ]
            )

        # Filter single atoms.
        train_ds = train_ds.filter(
            lambda element: tf.math.greater(tf.shape(element['atomic_numbers'])[0], tf.constant(1))
        )
        valid_ds = valid_ds.filter(
            lambda element: tf.math.greater(tf.shape(element['atomic_numbers'])[0], tf.constant(1))
        )
        if return_test:
            test_ds = test_ds.filter(
                lambda element: tf.math.greater(tf.shape(element['atomic_numbers'])[0], tf.constant(1))
            )

        # Create GraphTuples.
        train_ds = train_ds.map(
            lambda element: create_graph_tuple_tf(element, cutoff=cutoff)
        )
        valid_ds = valid_ds.map(
            lambda element: create_graph_tuple_tf(element, cutoff=cutoff)
        )
        if return_test:
            test_ds = test_ds.map(
                lambda element: create_graph_tuple_tf(element, cutoff=cutoff)
            )

        # Filter max forces.
        train_ds = train_ds.filter(
            lambda graph: tf.math.less(tf.math.reduce_max(graph.nodes['forces']), tf.constant(self.max_force_filter))
        )
        valid_ds = valid_ds.filter(
            lambda graph: tf.math.less(tf.math.reduce_max(graph.nodes['forces']), tf.constant(self.max_force_filter))
        )
        if return_test:
            test_ds = test_ds.filter(
                lambda graph: tf.math.less(tf.math.reduce_max(graph.nodes['forces']),
                                           tf.constant(self.max_force_filter))
            )

        if return_test:
            return train_ds, valid_ds, test_ds
        else:
            return train_ds, valid_ds
