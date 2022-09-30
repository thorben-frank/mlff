import jax
import numpy as np
import logging
import os

from typing import Dict
from mlff.src.indexing.indices import get_indices
from mlff.src.random.random import set_seeds
from mlff.src.io.io import read_json, save_dict, merge_dicts
from mlff.src.data.preprocessing import draw_strat_sample

# TODO: same split name for everything
class DataSet:
    def __init__(self, prop_keys: Dict, data: Dict):
        self.prop_keys = prop_keys
        self.data = data
        self.data, self.n_data = self._correct_shapes()
        self.splits = {}
        self.splits_info = {}

    def summary(self):
        raise NotImplementedError

    def _correct_shapes(self):
        # get n_data, kind of messy but best way I could think of
        q_data = {k: v for k, v in self.data.items()}

        def reshape(y):
            if len(y.shape) <= 1:
                return y.reshape(1, -1)
            else:
                return y

        q_data = jax.tree_map(lambda y: reshape(y), q_data)
        max_key = max(jax.tree_map(lambda y: len(y), q_data), key=jax.tree_map(lambda y: len(y), q_data).get)
        n_data = len(q_data[max_key])

        def repeat(name, y, repeats):
            if len(y) == 1:
                logging.info('Detected missing data dimension (0-th axis) for {}. Assume that the '
                             'data dimension is missing and repeat the entry {} times. Reshaped '
                             'array to ({}, {})'.format(name, n_data, n_data, y.shape[0]))
                return np.repeat(y, repeats=repeats, axis=0)
            elif 1 < len(y) < n_data:
                logging.info('Detected missing data dimension (0-th axis) for {}. Assume that the '
                             'data dimension is missing and repeat the entry {} times. Reshaped '
                             'array to ({}, {})'.format(name, n_data, n_data, y.shape))
                return np.repeat(y[None], repeats=repeats, axis=0)
            else:
                return y

        return {k: repeat(name=k, y=v, repeats=n_data) for (k, v) in q_data.items()}, n_data

    def neighborhood_list(self, r_cut):
        R_key = self.prop_keys.get('atomic_position')
        z_key = self.prop_keys.get('atomic_type')
        neigh_idx = get_indices(R=self.data[R_key], z=self.data[z_key], r_cut=r_cut)
        return neigh_idx

    def index_split(self,
                    data_idx_train,
                    data_idx_valid,
                    data_idx_test,
                    r_cut=None,
                    split_name='index_split'):
        d = {}
        for i, i_n in zip([data_idx_train, data_idx_valid, data_idx_test], ['train', 'valid', 'test']):
            _d = {}
            for k, v in self.data.items():
                if len(v.shape) <= 1:
                    logging.warning(
                        'Array with shape {} for quantity {} detected, such that we assume that the data dimension '
                        'is missing. Reshaped to ({}, {}).'.format(v.shape, k, len(i), v.shape[-1]))
                    v = np.repeat(v[None, :], repeats=len(i), axis=0)
                _d.update({k: v[i]})
                try:
                    _d.update({'{}_mean'.format(k): np.mean(v[i].reshape(-1))})
                    _d.update({'{}_scale'.format(k): np.std(v[i].reshape(-1))})
                except TypeError:
                    msg = 'TypeError detected when trying to calculate mean and std for quantity {}. No mean and ' \
                          'std calculated.'.format(k)
                    logging.info(msg)

                d.update({i_n: _d})

        if r_cut is not None:
            R_key = self.prop_keys.get('atomic_position')
            z_key = self.prop_keys.get('atomic_type')

            R_dat = np.concatenate([d['train'][R_key], d['valid'][R_key], d['test'][R_key]])
            z_dat = np.concatenate([d['train'][z_key], d['valid'][z_key], d['test'][z_key]])

            neigh_idxs = get_indices(R_dat, z_dat, r_cut=r_cut)
            n_train = len(data_idx_train)
            n_valid = len(data_idx_valid)

            idx_i_train, idx_i_valid, idx_i_test = np.split(neigh_idxs['idx_i'],
                                                            indices_or_sections=[n_train, n_train + n_valid])
            idx_j_train, idx_j_valid, idx_j_test = np.split(neigh_idxs['idx_j'],
                                                            indices_or_sections=[n_train, n_train + n_valid])

            d_idx = {'train': {'idx_i': idx_i_train, 'idx_j': idx_j_train},
                     'valid': {'idx_i': idx_i_valid, 'idx_j': idx_j_valid},
                     'test': {'idx_i': idx_i_test, 'idx_j': idx_j_test}}

            _d = {k: merge_dicts(v, d_idx[k])
                  for k, v in d.items() if k in ['train', 'valid', 'test']}

            d.update(_d)

        n_train = len(data_idx_train)
        n_valid = len(data_idx_valid)
        n_test = len(data_idx_test)
        self.splits.update({split_name: {'data_idx_train': np.array(data_idx_train),
                                         'data_idx_valid': np.array(data_idx_valid),
                                         'data_idx_test': np.array(data_idx_test)}})
        self.splits_info.update({split_name: {'n_train': n_train,
                                              'n_valid': n_valid,
                                              'n_test': n_test,
                                              'n_data': self.n_data,
                                              'r_cut': r_cut}})
        return d

    # TODO: allow to pass 0 for n_train and n_valid. Currently fails due to get_indices function.
    def random_split(self,
                     n_train,
                     n_valid,
                     n_test,
                     r_cut=None,
                     seed=0,
                     training=True,
                     split_name='random_split'):

        d = self.split_data(data=self.data,
                            n_train=n_train,
                            n_valid=n_valid,
                            n_test=n_test,
                            seed=seed)

        if r_cut is not None:
            R_key = self.prop_keys.get('atomic_position')
            z_key = self.prop_keys.get('atomic_type')

            if training:
                R_dat = np.concatenate([d['train'][R_key], d['valid'][R_key]])
                z_dat = np.concatenate([d['train'][z_key], d['valid'][z_key]])
            else:
                R_dat = np.concatenate([d['train'][R_key], d['valid'][R_key], d['test'][R_key]])
                z_dat = np.concatenate([d['train'][z_key], d['valid'][z_key], d['test'][z_key]])

            neigh_idxs = get_indices(R_dat, z_dat, r_cut=r_cut)
            idx_i_train, idx_i_valid, idx_i_test = np.split(neigh_idxs['idx_i'],
                                                            indices_or_sections=[n_train, n_train + n_valid])
            idx_j_train, idx_j_valid, idx_j_test = np.split(neigh_idxs['idx_j'],
                                                            indices_or_sections=[n_train, n_train + n_valid])

            d_idx = {'train': {'idx_i': idx_i_train, 'idx_j': idx_j_train},
                     'valid': {'idx_i': idx_i_valid, 'idx_j': idx_j_valid},
                     'test': {'idx_i': idx_i_test, 'idx_j': idx_j_test}}

            _d = {k: merge_dicts(v, d_idx[k])
                  for k, v in d.items() if k in ['train', 'valid', 'test']}

            d.update(_d)

        n_test = len(d['data_idx_test'])
        self.splits.update({split_name: {'data_idx_train': d['data_idx_train'],
                                         'data_idx_valid': d['data_idx_valid'],
                                         'data_idx_test': d['data_idx_test']}})
        self.splits_info.update({split_name: {'n_train': n_train,
                                              'n_valid': n_valid,
                                              'n_test': n_test,
                                              'n_data': self.n_data,
                                              'r_cut': r_cut,
                                              'seed': seed}})

        return d

    def strat_split(self,
                    n_train,
                    n_valid,
                    n_test,
                    strat_key,
                    r_cut=None,
                    seed=0,
                    training=True,
                    split_name='strat_split'):

        d = self.split_data(data=self.data,
                            n_train=n_train,
                            n_valid=n_valid,
                            n_test=n_test,
                            seed=seed,
                            draw_strat=strat_key)

        if r_cut is not None:
            R_key = self.prop_keys.get('atomic_position')
            z_key = self.prop_keys.get('atomic_type')

            if training:
                R_dat = np.concatenate([d['train'][R_key], d['valid'][R_key]])
                z_dat = np.concatenate([d['train'][z_key], d['valid'][z_key]])
            else:
                R_dat = np.concatenate([d['train'][R_key], d['valid'][R_key], d['test'][R_key]])
                z_dat = np.concatenate([d['train'][z_key], d['valid'][z_key], d['test'][z_key]])

            neigh_idxs = get_indices(R_dat, z_dat, r_cut=r_cut)
            idx_i_train, idx_i_valid, idx_i_test = np.split(neigh_idxs['idx_i'],
                                                            indices_or_sections=[n_train, n_train + n_valid])
            idx_j_train, idx_j_valid, idx_j_test = np.split(neigh_idxs['idx_j'],
                                                            indices_or_sections=[n_train, n_train + n_valid])

            d_idx = {'train': {'idx_i': idx_i_train, 'idx_j': idx_j_train},
                     'valid': {'idx_i': idx_i_valid, 'idx_j': idx_j_valid},
                     'test': {'idx_i': idx_i_test, 'idx_j': idx_j_test}}

            _d = {k: merge_dicts(v, d_idx[k])
                  for k, v in d.items() if k in ['train', 'valid', 'test']}

            d.update(_d)

        n_test = len(d['data_idx_test'])
        self.splits.update({split_name: {'data_idx_train': d['data_idx_train'],
                                         'data_idx_valid': d['data_idx_valid'],
                                         'data_idx_test': d['data_idx_test']}})
        self.splits_info.update({split_name: {'n_train': n_train,
                                              'n_valid': n_valid,
                                              'n_test': n_test,
                                              'n_data': self.n_data,
                                              'r_cut': r_cut,
                                              'seed': seed}})
        return d

    @staticmethod
    def split_data(data, n_train, n_valid, n_test=None, seed=0, draw_strat=None):
        set_seeds(seed)
        _k = list(data.keys())[0]
        n_data = len(data[_k])
        perm = np.random.RandomState(seed).permutation(n_data)
        idx_all = np.arange(n_data)[perm]

        if draw_strat:
            idx_train = draw_strat_sample(data[draw_strat][perm],
                                          n=n_train)
            idx_valid = np.array(list(set(idx_all) - set(idx_train)))
            # set sorts the indices, so we have to permute them again
            valid_perm = np.random.RandomState(seed).permutation(len(idx_valid))
            idx_valid = idx_valid[valid_perm][:n_valid]
        else:
            idx = idx_all[:n_train + n_valid]
            idx_train, idx_valid = np.split(idx, indices_or_sections=[n_train])

        # set sorts the indices, so we have to permute them again
        idx_test = np.array(list(set(idx_all) - set(idx_train) - set(idx_valid)))
        test_perm = np.random.RandomState(seed).permutation(len(idx_test))
        idx_test = idx_test[test_perm][:n_test]

        if n_test is None:
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
                if len(v.shape) <= 1:
                    logging.warning(
                        'Array with shape {} for quantity {} detected, such that we assume that the data dimension is '
                        'missing. Reshaped to ({}, {}).'.format(v.shape, k, len(i), v.shape[-1]))
                    v = np.repeat(v[None, :], repeats=n_data, axis=0)
                try:
                    _d.update({k: v[i]})
                except IndexError:
                    msg = 'Trying to access index {} for array under key {}, ' \
                          'which has shape {}, however.'.format(i, k, v.shape)
                    raise IndexError(msg)
                try:
                    _d.update({'{}_mean'.format(k): np.mean(v[i].reshape(-1))})
                    _d.update({'{}_scale'.format(k): np.std(v[i].reshape(-1))})
                except TypeError:
                    msg = 'TypeError detected when trying to calculate mean and std for quantity {}. No mean and std ' \
                          'calculated.'.format(k)
                    logging.info(msg)

                d.update({i_n: _d})
        return d

    def load_split(self, file, r_cut, split_name, n_train=None, n_valid=None, n_test=None):
        path, filename = os.path.split(file)
        split_idx = self.load_splits_from_file(path=path, filename=filename)[split_name]
        key_2_n = {'data_idx_train': n_train, 'data_idx_valid': n_valid, 'data_idx_test': n_test}

        valid_keys = list(key_2_n.keys())

        subset_split = {k: v[:key_2_n[k]] for (k, v) in split_idx.items() if k in valid_keys}
        return self.index_split(r_cut=r_cut, **subset_split)

    def save_splits_to_file(self, path, filename):
        splits_ = jax.tree_map(lambda y: y.tolist(), self.splits)
        save_dict(path=path, filename=filename, data=splits_, exists_ok=True)
        logging.info('Saved the data indices of the splits to {}'.format(os.path.join(path, filename)))

    @staticmethod
    def load_splits_from_file(path, filename):
        _data_idx = read_json(path=os.path.join(path, filename))
        return _data_idx

    def __dict_repr__(self):
        return {'dataset': self.splits_info}
