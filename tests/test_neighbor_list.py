from jax.config import config
config.update("jax_enable_x64", True)
import jax
import jax.numpy as jnp
import numpy as np
from ase import Atoms
from .test_data import load_data


def compare_distances(a, b, cutoff, atol=0.0, rtol=1e-7):
    a = filter_and_sort(a, cutoff)
    b = filter_and_sort(b, cutoff)

    np.testing.assert_allclose(a, b, atol=atol, rtol=rtol)


def filter_and_sort(distances, cutoff):
    d = distances[distances < cutoff]
    d = d[d > 0]
    return np.sort(d)


# TODO: test for non-orthorombic cells
def test_neighborhood_list():
    # load all data from the new ZrO2 data set
    data = load_data('test_solid.npz')

    zs = data['z']
    cells = data['unit_cell']
    Rs = data['R']

    cutoff = 4
    mic = False

    from mlff.indexing.indices import get_indices
    from tqdm import tqdm

    indices = get_indices(np.array(Rs), np.array(zs), r_cut=cutoff, cell=np.array(cells), mic=mic)

    for i in tqdm(range(len(Rs))):
        idx_i = indices['idx_i'][i]
        idx_j = indices['idx_j'][i]

        v_ij = Rs[i][idx_j] - Rs[i][idx_i]
        d_ij = jnp.linalg.norm(v_ij, axis=-1)

        atoms = Atoms(zs[i], positions=Rs[i])
        ase_d_ij = atoms.get_all_distances(mic=mic)
        compare_distances(d_ij, ase_d_ij, cutoff=cutoff)


def test_neighborhood_list_valid_mic():
    # load all data from the new ZrO2 data set
    data = load_data('test_solid.npz')
    zs = data['z']
    cells = data['unit_cell']
    Rs = data['R']

    cutoff = 4
    mic = True

    from mlff.cutoff_function import pbc_diff, add_cell_offsets
    from mlff.indexing.indices import get_pbc_indices, get_indices
    from tqdm import tqdm

    indices = get_indices(np.array(Rs), np.array(zs), r_cut=cutoff, cell=np.array(cells), mic=mic)
    if mic:
        pbc_indices = get_pbc_indices(np.array(Rs),
                                      np.array(cells),
                                      r_cut=cutoff,
                                      pbc=np.repeat(np.array([[True, True, True]]), axis=0, repeats=len(Rs))
                                      )

    for i in tqdm(range(len(Rs))):
        idx_i = indices['idx_i'][i]
        idx_j = indices['idx_j'][i]

        v_ij = Rs[i][idx_j] - Rs[i][idx_i]
        if mic:
            v_ij = jax.vmap(pbc_diff, in_axes=(0, None))(v_ij, cells[i])
        d_ij = jnp.linalg.norm(v_ij, axis=-1)

        atoms = Atoms(zs[i], positions=Rs[i], cell=cells[i], pbc=[True, True, True])
        ase_d_ij = atoms.get_all_distances(mic=mic)
        compare_distances(d_ij, ase_d_ij, cutoff=cutoff)

        if mic:
            pbc_idx_i = pbc_indices['idx_i'][i]
            pbc_idx_j = pbc_indices['idx_j'][i]
            shifts = pbc_indices['cell_offset'][i]

            pbc_v_ij = Rs[i][pbc_idx_j] - Rs[i][pbc_idx_i]
            pbc_v_ij = add_cell_offsets(pbc_v_ij, cell=cells[i], cell_offsets=shifts)
            pbc_d_ij = jnp.linalg.norm(pbc_v_ij, axis=-1)

            compare_distances(d_ij, pbc_d_ij, cutoff=np.inf)
            compare_distances(v_ij, pbc_v_ij, cutoff=np.inf)


def test_neighborhood_list_invalid_mic():
    # load all data from the new ZrO2 data set
    data = load_data('test_solid.npz')
    zs = data['z']
    cells = data['unit_cell']
    Rs = data['R']

    cutoff = 5
    mic = True

    from mlff.indexing.indices import get_pbc_indices, get_indices
    from tqdm import tqdm
    try:
        _ = get_indices(np.array(Rs), np.array(zs), r_cut=cutoff, cell=np.array(cells), mic=mic)
    except NotImplementedError:
        pass
    #
    # if mic:
    #     pbc_indices = get_pbc_indices(np.array(Rs),
    #                                   np.array(cells),
    #                                   r_cut=cutoff,
    #                                   pbc=np.repeat(np.array([[True, True, True]]), axis=0, repeats=len(Rs))
    #                                   )
    #
    # for i in tqdm(range(len(Rs))):
    #     atoms = Atoms(zs[i], positions=Rs[i], cell=cells[i], pbc=[True, True, True])
    #     ase_d_ij = atoms.get_all_distances(mic=mic)
    #
    #     if mic:
    #         pbc_idx_i = pbc_indices['idx_i'][i]
    #         pbc_idx_j = pbc_indices['idx_j'][i]
    #         shifts = pbc_indices['cell_offset'][i]
    #
    #         pbc_v_ij = Rs[i][pbc_idx_j] - Rs[i][pbc_idx_i]
    #         pbc_v_ij = add_cell_offsets(pbc_v_ij, cell=cells[i], cell_offsets=shifts)
    #         pbc_d_ij = jnp.linalg.norm(pbc_v_ij, axis=-1)
    #
    #         compare_distances(ase_d_ij, pbc_d_ij, cutoff=cutoff)
