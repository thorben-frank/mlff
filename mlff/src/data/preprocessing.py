import jax
import numpy as np
import jax.numpy as jnp
import logging
import os

from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from typing import (Any, Dict, Sequence, Tuple)

from mlff.src.random.random import set_seeds
from mlff.src.indexing.indices import get_indices

Array = Any


def split_data(data, quantities, n_train, n_valid, n_test=None, seed=0, draw_strat=None):
    set_seeds(seed)

    # get n_data, kind of messy but best way I could think of
    q_data = {k: v for k, v in data.items() if k in quantities}

    def reshape(y):
        if len(y.shape) <= 1:
            return y.reshape(-1, 1)
        else:
            return y

    q_data = jax.tree_map(lambda y: reshape(y), q_data)
    max_key = max(jax.tree_map(lambda y: len(y), q_data), key=jax.tree_map(lambda y: len(y), q_data).get)
    n_data = len(data[max_key])

    idx_all = np.arange(n_data)
    if draw_strat:
        idx = draw_strat_sample(data[draw_strat], n=n_train + n_valid)
    else:
        idx = idx_all[np.random.RandomState(seed+1).permutation(n_data)][:n_train+n_valid]

    idx_train, idx_valid = np.split(idx[np.random.RandomState(seed).permutation(len(idx))],
                                    indices_or_sections=[n_train])
    idx_test = np.array(list(set(idx_all) - set(idx_train) - set(idx_valid)))
    if n_test is not None:
        idx_test = idx_test[np.random.RandomState(seed+1).permutation(len(idx_test))][:n_test]
    else:
        n_test = len(idx_test)

    # assert no duplicates per subset
    assert len(set(idx_train)) == n_train
    assert len(set(idx_valid)) == n_valid
    assert len(set(idx_test)) == n_test
    # make sure there is no overlap
    assert len(set(idx_test) & set(idx_train)) == 0
    assert len(set(idx_test) & set(idx_valid)) == 0
    assert len(set(idx_train) & set(idx_valid)) == 0

    d = {'data_idx_train': idx_train, 'data_idx_valid': idx_valid, 'data_idx_test': idx_test}
    for i, i_n in zip([idx_train, idx_valid, idx_test], ['train', 'valid', 'test']):
        _d = {}
        for k, v in data.items():
            if k in quantities:
                if len(v.shape) <= 1:
                    logging.warning(
                        'Array with shape {} for quantity {} detected, such that we assume that the data dimension is '
                        'missing. Reshaped to ({}, {}).'.format(v.shape, k, len(i), v.shape[-1]))
                    v = np.repeat(v[None, :], repeats=n_data, axis=0)
                _d.update({k: v[i]})
                try:
                    _d.update({'{}_mean'.format(k): np.mean(v[i].reshape(-1))})
                    _d.update({'{}_scale'.format(k): np.std(v[i].reshape(-1))})
                except TypeError:
                    msg = 'TypeError detected when trying to calculate mean and std for quantity {}. No mean and std ' \
                          'calculated.'.format(k)
                    logging.info(msg)
            else:
                d.update({'{}'.format(k): v})
            d.update({i_n: _d})
    return d


def _get_quantity(data,
                  name: str,
                  n_train: int,
                  n_valid: int,
                  perm: Array):
    """
    Helper function for `split_data_to_file` function.

    Args:
        data ():
        name ():
        n_train ():
        n_valid ():
        perm ():

    Returns:

    """
    dat = data[name]
    n_data = len(perm)

    if (n_train+n_valid) > n_data:
        raise ValueError(
            "Number of training samples ({}) plus number of validation samples ({}) can not be larger than "
            "the total number of samples ({})".format(n_train, n_valid, n_data))

    new_ds = {}
    name_up = name.upper()
    scaler = StandardScaler()
    if dat.shape[0] == n_data:
        dat_train = dat[perm, ...][:int(n_train), ...].astype(dat.dtype)
        scaler.fit(dat_train.reshape(-1, 1))
        new_ds["{}_{}".format(name_up, 'train')] = dat_train
        new_ds["{}_{}".format(name_up, 'valid')] = dat[perm, ...][int(n_train):int(n_train+n_valid), ...].astype(dat.dtype)
        new_ds["{}_{}".format(name_up, 'test')] = dat[perm, ...][int(n_train+n_valid):, ...].astype(dat.dtype)
        new_ds["{}_{}".format(name_up, 'mean')] = scaler.mean_
        new_ds["{}_{}".format(name_up, 'scale')] = scaler.scale_
    else:
        if dat.ndim <= 1:
            dat = np.expand_dims(dat, axis=0)
        new_ds["{}_{}".format(name_up, 'train')] = np.repeat(dat, n_train, axis=0).astype(dat.dtype)
        new_ds["{}_{}".format(name_up, 'valid')] = np.repeat(dat, n_valid, axis=0).astype(dat.dtype)
        new_ds["{}_{}".format(name_up, 'test')] = np.repeat(dat, int(n_data-n_valid-n_train), axis=0).astype(dat.dtype)

    return new_ds


def split_data_to_file(file_path: str,
                       prefix: str,
                       quantities: Sequence[str],
                       n_train: int,
                       n_valid: int,
                       seed: Sequence[int],
                       save_path: str
                       ) -> None:
    """
    Split npz data that are in MD17 format into train, validation and training set given some seed and save it as a new
    npz file. The saved file will have the name '{prefix}_n_train_{n_train}_n_valid_{n_valid}_seed_{seed}.npz' and
    contains all keys from the original npz file. Keys that have been marked as quantities are split in such a way that
    for some key K, three new keys are generated which have the name K_train, K_valid, K_test. Note that all quantities
    are made capital letters, such that e.g. z in the original file would become Z_train, Z_valid, Z_test when split or
    just Z. Further it calculates the mean and standard deviation over all split quantities which are stored as keys
    K_mean and K_scale. If you want to use data processing functions from mlff package, please make sure to first
    process your data using this function.

    Args:
        file_path (str): Path to the original file.
        prefix (str): Prefix that is put at the beginning of the name filename of the saved file.
        quantities (List): List of quantities that should be split. For MD17 e.g. ['E', 'F', 'R', 'z']
        n_train (int): number of training points
        n_valid (int): number of validation points
        seed (int): seed that is used to generated the splits
        save_path (str): Path where to save the npz file.

    Returns:

    """

    data = np.load(file_path)

    shapes = []
    for q in quantities:
        shapes += [data[q].shape[0]]
    n_data = max(shapes)  # as there are some quantities which might only have a single row, e.g. 'z' in MD17 data

    ds = {}
    for s in seed:
        set_seeds(s)
        perm = np.random.permutation(n_data)
        for name, value in data.items():
            if name in quantities:
                ds.update(_get_quantity(data,
                                        name=name,
                                        n_train=n_train,
                                        n_valid=n_valid,
                                        perm=perm
                                        )
                          )
            else:
                d_ = {name.upper(): value}
                ds.update(d_)
        file_name = "{}_n_train_{}_n_valid_{}_seed_{}.npz".format(prefix, n_train, n_valid, s)
        print("Write file {}.npz into directory {}".format(file_name, save_path))
        np.savez(os.path.join(save_path, file_name), **ds)


def draw_strat_sample(T, n, excl_idxs=None):
    """
    Draw sample from dataset that preserves its original distribution.

    The distribution is estimated from a histogram were the bin size is
    determined using the Freedman-Diaconis rule. This rule is designed to
    minimize the difference between the area under the empirical
    probability distribution and the area under the theoretical
    probability distribution. A reduced histogram is then constructed by
    sampling uniformly in each bin. It is intended to populate all bins
    with at least one sample in the reduced histogram, even for small
    training sizes.

    Parameters
    ----------
        T : :obj:`numpy.ndarray`
            Dataset to sample from.
        n : int
            Number of examples.
        excl_idxs : :obj:`numpy.ndarray`, optional
            Array of indices to exclude from sample.

    Returns
    -------
        :obj:`numpy.ndarray`
            Array of indices that form the sample.
    """

    if excl_idxs is None or len(excl_idxs) == 0:
        excl_idxs = None

    if n == 0:
        return np.array([], dtype=np.uint)

    if T.size == n:  # TODO: this only works if excl_idxs=None
        assert excl_idxs is None
        return np.arange(n)

    if n == 1:
        idxs_all_non_excl = np.setdiff1d(
            np.arange(T.size), excl_idxs, assume_unique=True
        )
        return np.array([np.random.choice(idxs_all_non_excl)])

    # Freedman-Diaconis rule
    h = 2 * np.subtract(*np.percentile(T, [75, 25])) / np.cbrt(n)
    n_bins = int(np.ceil((np.max(T) - np.min(T)) / h)) if h > 0 else 1
    n_bins = min(
        n_bins, int(n / 2)
    )  # Limit number of bins to half of requested subset size.

    bins = np.linspace(np.min(T), np.max(T), n_bins, endpoint=False)
    idxs = np.digitize(T, bins)

    # Exclude restricted indices.
    if excl_idxs is not None and excl_idxs.size > 0:
        idxs[excl_idxs] = n_bins + 1  # Impossible bin.

    uniq_all, cnts_all = np.unique(idxs, return_counts=True)

    # Remove restricted bin.
    if excl_idxs is not None and excl_idxs.size > 0:
        excl_bin_idx = np.where(uniq_all == n_bins + 1)
        cnts_all = np.delete(cnts_all, excl_bin_idx)
        uniq_all = np.delete(uniq_all, excl_bin_idx)

    # Compute reduced bin counts.
    reduced_cnts = np.ceil(cnts_all / np.sum(cnts_all, dtype=float) * n).astype(int)
    reduced_cnts = np.minimum(
        reduced_cnts, cnts_all
    )  # limit reduced_cnts to what is available in cnts_all

    # Reduce/increase bin counts to desired total number of points.
    reduced_cnts_delta = n - np.sum(reduced_cnts)

    while np.abs(reduced_cnts_delta) > 0:
        # How many members can we remove from an arbitrary bucket, without any bucket with more than one member going to zero?
        max_bin_reduction = np.min(reduced_cnts[np.where(reduced_cnts > 1)]) - 1

        # Generate additional bin members to fill up/drain bucket counts of subset. This array contains (repeated) bucket IDs.
        outstanding = np.random.choice(
            uniq_all,
            min(max_bin_reduction, np.abs(reduced_cnts_delta)),
            p=(reduced_cnts - 1) / np.sum(reduced_cnts - 1, dtype=float),
            replace=True,
        )
        uniq_outstanding, cnts_outstanding = np.unique(
            outstanding, return_counts=True
        )  # Aggregate bucket IDs.

        outstanding_bucket_idx = np.where(
            np.in1d(uniq_all, uniq_outstanding, assume_unique=True)
        )[
            0
        ]  # Bucket IDs to Idxs.
        reduced_cnts[outstanding_bucket_idx] += (
                np.sign(reduced_cnts_delta) * cnts_outstanding
        )
        reduced_cnts_delta = n - np.sum(reduced_cnts)

    # Draw examples for each bin.
    idxs_train = np.empty((0,), dtype=int)
    for uniq_idx, bin_cnt in zip(uniq_all, reduced_cnts):
        idx_in_bin_all = np.where(idxs.ravel() == uniq_idx)[0]
        idxs_train = np.append(
            idxs_train, np.random.choice(idx_in_bin_all, bin_cnt, replace=False)
        )

    return idxs_train


def get_per_atom_shift(z: Array, q: Array, pad_value: str = None) -> Tuple[Array, Array]:
    """
    Get per atom shift, given the atomic numbers across structures. The per atom shift is calculated by first constructing
    an matrix that counts the occurrences of each atomic type for each structure which gives a matrix `A` of shape
    (n_data, max_z+1), where `max_z` is the largest atomic number. The per atom shifts are then calculated by solving
    the linear equation `A @ x = q`. Here, `q` is the target quantity of shape (n_data). The function returns the
    shifts as a vector `shifts` of shape (max_z + 1), where e.g. the shift for carbon can be accessed by shifts[6], or
    for hydrogen by shifts[1]. The extra dimension comes from the fact that we want a one-to-one correspondence between
    index and atomic type. It also returns the rescaled quantities `q` using `shifts`. If one has differently sized
    structures, `z` has to be padded in order to solve the linear system,

    Args:
        z (Array): Atomic numbers, shape: (n_data, n)
        q (Array): Quantity to fit the shifts to, shape: (n_data)
        pad_value (int): If data has been padded, what is the value used for padding

    Returns: Tuple[Array]

    """
    u, _ = np.unique(z, return_counts=True)

    if pad_value is not None:
        idx_ = (u != pad_value)
    else:
        idx_ = np.arange(len(u))

    count_fn = lambda y: dict(zip(*np.unique(y, return_counts=True)))
    lhs_counts = list(map(count_fn, z))

    v = DictVectorizer(sparse=False)
    X = v.fit_transform(lhs_counts)
    X = X[..., idx_]

    sol = np.linalg.lstsq(X, q)

    shifts = np.zeros(np.max(u) + 1)
    for k, v in dict(zip(u[idx_], sol[0])).items():
        shifts[k] = v

    v_take = jax.vmap(jnp.take, in_axes=(None, 0))
    q_scaled = q - v_take(shifts, z).sum(axis=-1)
    return shifts, q_scaled
