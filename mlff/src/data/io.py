# TODO: legacy code, remove at some point


import numpy as np
from typing import (Callable, Dict)

from mlff.src.indexing.indices import get_indices


def get_training_data(file_path: str,
                      scale=True,
                      scale_target='energy') -> (Dict, Dict):
    train_ds, valid_ds, _ = get_data(file_path=file_path, scale=scale, scale_target=scale_target)
    return train_ds, valid_ds


def get_test_data(file_path: str,
                  n_test: int) -> Dict:
    _, _, test_ds = get_data(file_path=file_path, n_test=n_test)
    return test_ds


def get_data(file_path: str,
             scale=True,
             scale_target='energy',
             n_test: int = -1
             ) -> (Dict, Dict, Dict, Callable, Callable):

    data = np.load(file_path)
    E_mean = data['E_mean']
    E_scale = data['E_scale']
    F_scale = data['F_scale']

    if scale_target == 'energy':
        energy_scaler = lambda x: (x-E_mean)/E_scale if scale else x
        force_scaler = lambda x: x/E_scale if scale else x
    elif scale_target == 'force':
        energy_scaler = lambda x: (x - E_mean) / F_scale if scale else x
        force_scaler = lambda x: x / F_scale if scale else x

    train_ds = {'R': data['R_train'],
                'E': energy_scaler(data['E_train']),
                'F': force_scaler(data['F_train']),
                'z': data['Z_train']
                }

    valid_ds = {'R': data['R_valid'],
                'E': energy_scaler(data['E_valid']),
                'F': force_scaler(data['F_valid']),
                'z': data['Z_valid']
                }

    test_ds = {'R': data['R_test'][:n_test, ...],
               'E': data['E_test'][:n_test, ...],
               'F': data['F_test'][:n_test, ...],
               'z': data['Z_test'][:n_test, ...]
               }

    return train_ds, valid_ds, test_ds


def get_training_data_indexed(file_path: str,
                              r_cut: float,
                              scale=True,
                              scale_target='energy') -> (Dict, Dict):

    train_ds, valid_ds, _ = get_data_indexed(file_path=file_path,
                                             r_cut=r_cut,
                                             scale=scale,
                                             scale_target=scale_target,
                                             n_test=0)
    return train_ds, valid_ds


def get_test_data_indexed(file_path: str,
                          r_cut: float,
                          n_test: int) -> Dict:

    _, _, test_ds = get_data_indexed(file_path=file_path,
                                     r_cut=r_cut,
                                     n_test=n_test)
    return test_ds


def get_data_indexed(file_path: str,
                     r_cut: float,
                     scale=True,
                     scale_target='energy',
                     n_test: int = None
                     ) -> (Dict, Dict, Dict, Callable, Callable):

    data = np.load(file_path)
    E_mean = data['E_mean']
    E_scale = data['E_scale']
    F_scale = data['F_scale']

    n_train = len(data['E_train'])
    n_valid = len(data['E_valid'])
    n_test = len(data['E_test']) if n_test is None else n_test

    if scale_target == 'energy':
        energy_scaler = lambda x: (x-E_mean)/E_scale if scale else x
        force_scaler = lambda x: x/E_scale if scale else x
    elif scale_target == 'force':
        energy_scaler = lambda x: (x - E_mean) / F_scale if scale else x
        force_scaler = lambda x: x / F_scale if scale else x

    R = np.concatenate([data['R_train'], data['R_valid'], data['R_test'][:n_test]], axis=0)
    z = np.concatenate([data['Z_train'], data['Z_valid'], data['Z_test'][:n_test]], axis=0)
    d_idx = get_indices(R, z, r_cut=r_cut)
    idx_i = d_idx['idx_i']
    idx_j = d_idx['idx_j']
    idx_i_train, idx_i_valid, idx_i_test, _ = np.split(idx_i, [n_train, n_train+n_valid, n_train+n_valid+n_test])
    idx_j_train, idx_j_valid, idx_j_test, _ = np.split(idx_j, [n_train, n_train+n_valid, n_train+n_valid+n_test])

    train_ds = {'R': data['R_train'],
                'E': energy_scaler(data['E_train']),
                'F': force_scaler(data['F_train']),
                'z': data['Z_train'],
                'idx_i': idx_i_train,
                'idx_j': idx_j_train
                }

    valid_ds = {'R': data['R_valid'],
                'E': energy_scaler(data['E_valid']),
                'F': force_scaler(data['F_valid']),
                'z': data['Z_valid'],
                'idx_i': idx_i_valid,
                'idx_j': idx_j_valid
                }

    test_ds = {'R': data['R_test'][:n_test, ...],
               'E': data['E_test'][:n_test, ...],
               'F': data['F_test'][:n_test, ...],
               'z': data['Z_test'][:n_test, ...],
               'idx_i': idx_i_test,
               'idx_j': idx_j_test
               }

    return train_ds, valid_ds, test_ds


def get_scaler(file_path: str, scale_target='force') -> (Callable, Callable):
    data = np.load(file_path)
    E_mean = data['E_mean']
    E_scale = data['E_scale']
    F_scale = data['F_scale']
    if scale_target == 'energy':
        energy_scaler = lambda x: (x - E_mean) / E_scale
        force_scaler = lambda x: x / E_scale
    elif scale_target == 'force':
        energy_scaler = lambda x: (x - E_mean) / F_scale
        force_scaler = lambda x: x / F_scale

    return energy_scaler, force_scaler


def get_inverse_scaler(file_path: str, scale_target='force') -> (Callable, Callable):

    data = np.load(file_path)
    E_mean = data['E_mean']
    E_scale = data['E_scale']
    F_scale = data['F_scale']
    if scale_target == 'energy':
        energy_scaler_inv = lambda x: x * E_scale + E_mean
        force_scaler_inv = lambda x: x * E_scale
    elif scale_target == 'force':
        energy_scaler_inv = lambda x: x * F_scale + E_mean
        force_scaler_inv = lambda x: x * F_scale
    return energy_scaler_inv, force_scaler_inv
