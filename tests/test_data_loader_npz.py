import numpy as np
import jraph
import numpy.testing as npt
from mlff.data import NpzDataLoaderSparse
import pkg_resources
import pytest


@pytest.mark.parametrize("calculate_neighbors_lr", [True, False])
def test_data_load(calculate_neighbors_lr: bool):
    filename = 'test_data/ethanol.npz'
    f = pkg_resources.resource_filename(__name__, filename)

    loader = NpzDataLoaderSparse(input_file=f)
    all_data, data_stats = loader.load(
        cutoff=5.,
        calculate_neighbors_lr=calculate_neighbors_lr,
        cutoff_lr=10.
    )
    npt.assert_equal(len(all_data), 2000)
    npt.assert_equal(loader.cardinality(), 2000)

    npt.assert_equal(data_stats['max_num_of_nodes'], 9)
    npt.assert_equal(data_stats['max_num_of_edges'], 72)

    npt.assert_(isinstance(all_data[0], jraph.GraphsTuple))

    for i in range(0, 100)[::10]:
        forces = all_data[i].nodes.get('forces')
        positions = all_data[i].nodes.get('positions')
        atomic_numbers = all_data[i].nodes.get('atomic_numbers')
        hirshfeld_ratios = all_data[i].nodes['hirshfeld_ratios']
        senders = all_data[i].senders
        receivers = all_data[i].receivers
        energy = all_data[i].globals.get('energy')
        cell = all_data[i].edges.get('cell')
        cell_offset = all_data[i].edges.get('cell_offset')
        stress = all_data[i].globals['stress']
        total_charge = all_data[i].globals['total_charge']
        num_unpaired_electrons = all_data[i].globals['num_unpaired_electrons']
        dipole_vec = all_data[i].globals['dipole_vec']

        num_pairs = all_data[i].n_pairs
        idx_i_lr = all_data[i].idx_i_lr
        idx_j_lr = all_data[i].idx_j_lr

        npt.assert_(isinstance(forces, np.ndarray))
        npt.assert_(isinstance(energy, np.ndarray))
        npt.assert_(isinstance(atomic_numbers, np.ndarray))
        npt.assert_(isinstance(senders, np.ndarray))
        npt.assert_(isinstance(receivers, np.ndarray))
        npt.assert_(isinstance(idx_i_lr, np.ndarray))
        npt.assert_(isinstance(idx_j_lr, np.ndarray))

        npt.assert_equal(forces.dtype, np.float64)
        npt.assert_equal(energy.dtype, np.float64)
        npt.assert_equal(positions.dtype, np.float64)
        npt.assert_equal(atomic_numbers.dtype, np.int64)
        npt.assert_equal(senders.dtype, np.int64)
        npt.assert_equal(receivers.dtype, np.int64)
        npt.assert_equal(total_charge.dtype, np.int16)
        npt.assert_equal(num_unpaired_electrons.dtype, np.int16)

        npt.assert_equal(cell, None)
        npt.assert_equal(cell_offset, None)

        stress_expected = np.empty((1, 6))
        stress_expected[:] = np.nan
        npt.assert_equal(stress, stress_expected)

        num_atoms = 9
        hirshfeld_ratios_expected = np.empty((num_atoms,))
        hirshfeld_ratios_expected[:] = np.nan
        npt.assert_equal(hirshfeld_ratios, hirshfeld_ratios_expected)

        dipole_vec_expected = np.empty((1, 3))
        dipole_vec_expected[:] = np.nan
        npt.assert_equal(dipole_vec, dipole_vec_expected)

        npt.assert_equal(atomic_numbers.shape, (num_atoms,))
        npt.assert_equal(positions.shape, (num_atoms, 3))
        npt.assert_equal(energy.shape, (1,))
        npt.assert_equal(forces.shape, (num_atoms, 3))
        npt.assert_equal(total_charge.shape, (1,))
        npt.assert_equal(total_charge, 0)
        npt.assert_equal(num_unpaired_electrons.shape, (1,))
        npt.assert_equal(num_unpaired_electrons, 0)
        npt.assert_equal(len(senders), len(receivers))
        npt.assert_equal(len(senders), 72)

        npt.assert_equal(len(idx_j_lr), len(idx_j_lr))
        if calculate_neighbors_lr:
            npt.assert_equal(num_pairs, num_atoms * num_atoms - num_atoms)
            npt.assert_equal(len(idx_i_lr), num_pairs)
            npt.assert_equal(len(idx_j_lr), num_pairs)
            npt.assert_equal(senders, idx_j_lr)
            npt.assert_equal(receivers, idx_i_lr)
        else:
            npt.assert_equal(idx_i_lr, np.array([]).reshape(-1))
            npt.assert_equal(idx_j_lr, np.array([]).reshape(-1))
            npt.assert_equal(num_pairs, 0)


@pytest.mark.parametrize("calculate_neighbors_lr", [True, False])
def test_data_load_with_pbc(calculate_neighbors_lr: bool):
    filename = 'test_data/data_set_pbc.npz'
    f = pkg_resources.resource_filename(__name__, filename)
    loader = NpzDataLoaderSparse(input_file=f)
    if calculate_neighbors_lr:
        with npt.assert_raises(NotImplementedError):
            out = loader.load(
                cutoff=4.,
                calculate_neighbors_lr=calculate_neighbors_lr,
                cutoff_lr=75.
            )
    else:
        all_data, data_stats = loader.load(
            cutoff=4.,
            calculate_neighbors_lr=calculate_neighbors_lr,
            cutoff_lr=75.
        )
        npt.assert_equal(len(all_data), 50)
        npt.assert_equal(data_stats['max_num_of_nodes'], 110)

        npt.assert_(isinstance(all_data[0], jraph.GraphsTuple))

        for i in range(0, 50)[::5]:
            positions = all_data[i].nodes.get('positions')
            atomic_numbers = all_data[i].nodes.get('atomic_numbers')
            senders = all_data[i].senders
            receivers = all_data[i].receivers
            cell = all_data[i].edges.get('cell')
            cell_offset = all_data[i].edges.get('cell_offset')
            energy = all_data[i].globals.get('energy')
            forces = all_data[i].nodes.get('forces')
            stress = all_data[i].globals['stress']

            npt.assert_(isinstance(energy, np.ndarray))
            npt.assert_(isinstance(forces, np.ndarray))
            npt.assert_(isinstance(cell, np.ndarray))
            npt.assert_(isinstance(cell_offset, np.ndarray))
            npt.assert_(isinstance(atomic_numbers, np.ndarray))
            npt.assert_(isinstance(senders, np.ndarray))
            npt.assert_(isinstance(receivers, np.ndarray))

            npt.assert_equal(positions.dtype, np.float64)
            npt.assert_equal(atomic_numbers.dtype, np.int64)
            npt.assert_equal(energy.dtype, np.float64)
            npt.assert_equal(forces.dtype, np.float64)
            npt.assert_equal(senders.dtype, np.int64)
            npt.assert_equal(receivers.dtype, np.int64)
            npt.assert_equal(cell.dtype, np.float64)
            npt.assert_equal(cell_offset.dtype, np.int64)

            stress_expected = np.empty((1, 6))
            stress_expected[:] = np.nan
            npt.assert_equal(stress, stress_expected)

            num_atoms = 110
            npt.assert_equal(atomic_numbers.shape, (num_atoms,))
            npt.assert_equal(positions.shape, (num_atoms, 3))
            npt.assert_equal(energy.shape, (1,))
            npt.assert_equal(forces.shape, (num_atoms, 3))
            npt.assert_equal(cell.shape, (len(senders), 3, 3))
            npt.assert_equal(len(senders), len(receivers))
            npt.assert_equal(cell_offset.shape, (len(senders), 3))


@pytest.mark.parametrize("calculate_neighbors_lr", [True, False])
def test_jraph_dynamically_batch(calculate_neighbors_lr: bool):
    """
    Loaded data is compatible with jraph.dynamically_batch(...).

    Args:
        calculate_neighbors_lr ():

    Returns:

    """
    filename = 'test_data/ethanol.npz'
    f = pkg_resources.resource_filename(__name__, filename)

    loader = NpzDataLoaderSparse(input_file=f)
    all_data, data_stats = loader.load(
        cutoff=4.,
        calculate_neighbors_lr=calculate_neighbors_lr,
        cutoff_lr=75.
    )

    num_atoms = 9

    if calculate_neighbors_lr is False:
        for x in jraph.dynamically_batch(
            all_data,
            n_node=3 * num_atoms + 1,
            n_edge=3 * data_stats['max_num_of_edges'] + 1,
            n_graph=10,
            n_pairs=0
        ):
            npt.assert_equal(
                x.n_pairs,
                np.zeros(len(jraph.get_graph_padding_mask(x)))
            )
            npt.assert_equal(x.idx_i_lr.shape, (0, ))
            npt.assert_equal(x.idx_j_lr.shape, (0, ))
    else:
        # To small n_pairs
        with npt.assert_raises(RuntimeError):
            for x in jraph.dynamically_batch(
                all_data,
                n_node=3 * num_atoms + 1,
                n_edge=3 * data_stats['max_num_of_edges'] + 1,
                n_graph=10,
                n_pairs=5
            ):
                pass

        for k in [1, 5, 9]:
            # n_node and n_edge have space for 9 graphs so n_pairs determines maximum.
            for x in jraph.dynamically_batch(
                    all_data[:k*3],  # always do three batches
                    n_node=9 * num_atoms + 1,
                    n_edge=9 * data_stats['max_num_of_edges'] + 1,
                    n_graph=10,
                    n_pairs=k * (num_atoms * num_atoms - num_atoms) + 1
            ):
                npt.assert_equal(np.sum(jraph.get_graph_padding_mask(x)), np.array([k]))
