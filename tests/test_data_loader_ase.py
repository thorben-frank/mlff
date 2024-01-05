import jax.numpy as jnp
import jraph
import numpy.testing as npt
from mlff.data import AseDataLoaderSparse
import pkg_resources


def test_data_load_no_pbc():
    filename = 'test_data/data_set.xyz'
    f = pkg_resources.resource_filename(__name__, filename)

    loader = AseDataLoaderSparse(input_file=f)
    all_data = loader.load_all(cutoff=4.)
    npt.assert_equal(len(all_data), 100)

    npt.assert_(isinstance(all_data[0], jraph.GraphsTuple))

    for i in range(0, 100)[::10]:
        forces = all_data[i].nodes.get('forces')
        positions = all_data[i].nodes.get('positions')
        atomic_numbers = all_data[i].nodes.get('atomic_numbers')
        senders = all_data[i].senders
        receivers = all_data[i].receivers
        energy = all_data[i].globals.get('energy')
        cell = all_data[i].edges['cell']
        cell_offset = all_data[i].edges['cell_offset']
        stress = all_data[i].globals['stress']

        npt.assert_(isinstance(forces, jnp.ndarray))
        npt.assert_(isinstance(energy, jnp.ndarray))
        npt.assert_(isinstance(atomic_numbers, jnp.ndarray))
        npt.assert_(isinstance(senders, jnp.ndarray))
        npt.assert_(isinstance(receivers, jnp.ndarray))

        npt.assert_equal(forces.dtype, jnp.float32)
        npt.assert_equal(energy.dtype, jnp.float32)
        npt.assert_equal(positions.dtype, jnp.float32)
        npt.assert_equal(atomic_numbers.dtype, jnp.int32)
        npt.assert_equal(senders.dtype, jnp.int32)
        npt.assert_equal(receivers.dtype, jnp.int32)

        npt.assert_equal(cell, None)
        npt.assert_equal(cell_offset, None)
        npt.assert_equal(stress, None)

        num_atoms = 24
        npt.assert_equal(atomic_numbers.shape, (num_atoms,))
        npt.assert_equal(positions.shape, (num_atoms, 3))
        npt.assert_equal(energy.shape, (1,))
        npt.assert_equal(forces.shape, (num_atoms, 3))
        npt.assert_equal(len(senders), len(receivers))


def test_data_load_with_pbc():
    filename = 'test_data/data_set_pbc.xyz'
    f = pkg_resources.resource_filename(__name__, filename)

    loader = AseDataLoaderSparse(input_file=f)
    all_data = loader.load_all(cutoff=4.)
    npt.assert_equal(len(all_data), 50)

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

        npt.assert_(isinstance(energy, jnp.ndarray))
        npt.assert_(isinstance(forces, jnp.ndarray))
        npt.assert_(isinstance(cell, jnp.ndarray))
        npt.assert_(isinstance(cell_offset, jnp.ndarray))
        npt.assert_(isinstance(atomic_numbers, jnp.ndarray))
        npt.assert_(isinstance(senders, jnp.ndarray))
        npt.assert_(isinstance(receivers, jnp.ndarray))

        npt.assert_equal(positions.dtype, jnp.float32)
        npt.assert_equal(atomic_numbers.dtype, jnp.int32)
        npt.assert_equal(energy.dtype, jnp.float32)
        npt.assert_equal(forces.dtype, jnp.float32)
        npt.assert_equal(senders.dtype, jnp.int32)
        npt.assert_equal(receivers.dtype, jnp.int32)
        npt.assert_equal(cell.dtype, jnp.float32)
        npt.assert_equal(cell_offset.dtype, jnp.int32)

        npt.assert_equal(stress, None)

        num_atoms = 192
        npt.assert_equal(atomic_numbers.shape, (num_atoms,))
        npt.assert_equal(positions.shape, (num_atoms, 3))
        npt.assert_equal(energy.shape, (1,))
        npt.assert_equal(forces.shape, (num_atoms, 3))
        npt.assert_equal(cell.shape, (3, 3))
        npt.assert_equal(len(senders), len(receivers))
        npt.assert_equal(cell_offset.shape, (len(senders), 3))
