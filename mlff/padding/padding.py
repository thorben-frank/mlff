import jax

import numpy as np
from jax.ops import segment_sum
from functools import partial
from mlff.geometric.metric import coordinates_to_distance_matrix, coordinates_to_distance_matrix_mic


def index_padding_length(R, z, r_cut, mic, cell):
    """
        For the coordinates of the same molecule, determine the padding length for each of the `n_data` frames given some
        cutoff radius `r_cut`. As atoms may leave or enter the neighborhood for a given atom, one can have different lengths
        for the index lists even for the same molecule. The suggested padding length is the difference between the
        maximal number of indices over the whole training set and the number of indices for each frame.

        Args:
            R (Array): Atomic coordinates, shape: (n_data,n,3)
            z (Array): Atomic types, shape: (n_data,n)
            r_cut (float): Cutoff distance

        Returns: Padding lengths, shape: (n_data)

        """
    if mic:
        distance_fn = lambda r, c: coordinates_to_distance_matrix_mic(r, c)
    else:
        distance_fn = lambda r, *args, **kwargs: coordinates_to_distance_matrix(r)
        cell = np.zeros(len(R))

    # TODO: batching
    distance_fn = jax.jit(jax.vmap(distance_fn))

    n = R.shape[-2]
    n_data = R.shape[0]
    idx = np.indices((n_data, n, n))
    msk_ij = (np.einsum('...i, ...j -> ...ij', z, z) != 0).astype(np.int16)
    Dij = distance_fn(R, cell).squeeze()
    idx_seg, _, _ = np.split(idx[:, np.where((msk_ij * Dij <= r_cut) & (msk_ij * Dij > 0), True, False)],
                             indices_or_sections=3,
                             axis=0)

    segment_length = segment_sum(np.ones(len(idx_seg), ), segment_ids=idx_seg)
    pad_length = np.array((max(segment_length) - segment_length))
    return pad_length


def pad_atomic_types(z, n_max, pad_value=0):
    n = z.shape[-1]

    pad_length = n_max - n
    assert pad_length >= 0

    return np.pad(z, ((0, 0), (0, pad_length)), mode='constant', constant_values=((0, 0), (0, pad_value)))


def pad_coordinates(R, n_max, pad_value=0):
    n = R.shape[-2]

    pad_length = n_max - n
    assert pad_length >= 0

    return np.pad(R, ((0, 0), (0, pad_length), (0, 0)), mode='constant',
                  constant_values=((0, 0), (0, 0), (0, pad_value)))


def pad_forces(F, n_max, pad_value=0):
    """
    Padding of the atomic forces. Takes input arrays with shape (B,n,3).

    Args:
        F (Array): Array of atomic forces, shape: (B,n,3)
        n_max (int): Target length.
        pad_value (float): Value used for padding, defaults to 0.

    Returns: New array with padded forces, shape: (B,n_max,3)

    """
    n = F.shape[-2]

    pad_length = n_max - n
    assert pad_length >= 0

    return np.pad(F, ((0, 0), (0, pad_length), (0, 0)), mode='constant',
                  constant_values=((0, 0), (0, 0), (0, pad_value)))


def pad_per_atom_quantity(q, n_max, pad_value=0):
    n = q.shape[-2]

    pad_length = n_max - n
    assert pad_length >= 0

    return np.pad(q, ((0, 0), (0, pad_length), (0, 0)), mode='constant',
                  constant_values=((0, 0), (0, 0), (0, pad_value)))


def pad_indices(idx_i, idx_j, n_pair_max, pad_value=-1):
    n_pair = idx_i.shape[-1]
    assert idx_j.shape[-1] == n_pair

    pad_length = n_pair_max - n_pair
    assert pad_length >= 0

    pad = partial(np.pad, pad_width=((0, 0), (0, pad_length)), mode='constant',
                  constant_values=((0, 0), (0, pad_value)))
    pad_idx_i, pad_idx_j = map(pad, [idx_i, idx_j])
    return pad_idx_i, pad_idx_j
