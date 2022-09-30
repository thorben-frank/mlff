import jax
import jax.numpy as jnp
import numpy as np
import logging

from functools import partial
from typing import Tuple
from tqdm import tqdm

from mlff.src.geometric.metric import coordinates_to_distance_matrix
from mlff.src.padding.padding import index_padding_length


def non_diagonal_indices(n: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
        Get all non-diagonal indices.

        Args:
            n (int): Number of atoms.

        Returns: Tuple of all pairwise indices, shape: Tuple[(n_all_pairs), (n_all_pairs)]

    """
    up_i, up_j = jnp.triu_indices(n, k=1)
    lo_i, lo_j = jnp.tril_indices(n, k=-1)
    idx_i = jnp.concatenate((up_i, lo_i))
    idx_j = jnp.concatenate((up_j, lo_j))
    return idx_i, idx_j


def get_indices_from_mask(n: int, square_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
        Get indices from a square mask. Generally not jit'able, so rather use for preprocessing. E.g. extracting
        the indices if one has some physically meaningful interaction mask.

        Args:
            n (int): Number of atoms
            square_mask (Array): Boolean square matrix representation for index selection, shape: (n,n)

        Returns: Tuple of all pairwise indices, shape: Tuple[(n_all_pairs), (n_all_pairs)]

    """
    idx = jnp.indices((n, n))
    _square_mask = square_mask
    _square_mask[jnp.diag_indices(n)] = False
    idx_ = idx[:, np.where(_square_mask, True, False)]  # shape: (2,n_all_pairs)
    squeeze = partial(np.squeeze, axis=0)
    idx_i, idx_j = map(squeeze, np.split(idx_, indices_or_sections=2, axis=0))
    return idx_i, idx_j


def get_indices(R: np.ndarray, z: np.ndarray, r_cut: float):
    """
        For the `n_data` data frames, return index lists for centering and neighboring atoms given some cutoff radius
        `r_cut` for each structure. As atoms may leave or enter the neighborhood for a given atom within the dataset,
        one can have different lengths for the index lists, even for same structures. Thus, the index lists are padded
        wrt the length `n_pairs_max` of the longest index list observed over all frames in the coordinates `R` across
        all structures. Note, that this is suboptimal if one has a wide range number of atoms in the same dataset. The
        coordinates `R` and the atomic types `z` can also be already padded (see `mlff.src.padding.padding`) and are
        assumed to be padded with `0` if padded. Index values are padded with `-1`.

        Args:
            R (Array): Atomic coordinates, shape: (n_data,n,3)
            z (Array): Atomic types, shape: (n_data, n)
            r_cut (float): Cutoff distance

        Returns: Tuple of centering and neighboring indices, shape: Tuple[(n_pairs_max), (n_pairs_max)]

        """

    n = R.shape[-2]
    n_data = R.shape[0]
    pad_length = index_padding_length(R, z, r_cut)
    idx = np.indices((n, n))

    def get_idx(i):
        Dij = coordinates_to_distance_matrix(R[i]).squeeze(axis=-1)  # shape: (n,n)
        msk_ij = (np.einsum('i, j -> ij', z[i], z[i]) != 0).astype(np.int16)  # shape: (n,n)
        Dij_x_msk_ij = Dij * msk_ij  # shape: (n,n)
        idx_ = idx[:, np.where((Dij_x_msk_ij <= r_cut) & (Dij_x_msk_ij > 0), True, False)]  # shape: (2,n_pairs)
        pad_idx = np.pad(idx_, ((0, 0), (0, int(pad_length[i]))), mode='constant', constant_values=((0, 0), (0, -1)))
        # shape: (2,n_pair+pad_length)
        return pad_idx

    logging.info('Generate neighborhood lists for {} geometries: '.format(n_data))
    pad_idx_i, pad_idx_j = map(np.squeeze,
                               np.split(np.array(list(map(get_idx, tqdm(range(n_data))))),
                                        indices_or_sections=2,
                                        axis=-2))

    return {'idx_i': pad_idx_i, 'idx_j': pad_idx_j}
