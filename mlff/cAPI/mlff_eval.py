import argparse
import logging
import os
from functools import partial
from pprint import pprint
from typing import Dict
import json

import jax
import numpy as np
from ase.units import *
from pathlib import Path

from mlff.cAPI.process_argparse import StoreDictKeyPair
from mlff.data import DataSet, DataTuple
from mlff.inference.evaluation import evaluate_model, mae_metric, rmse_metric, r2_metric
from mlff.io import read_json, load_params_from_ckpt_dir
from mlff.nn.stacknet import (
    get_energy_force_stress_fn,
    get_obs_and_force_fn,
    get_observable_fn,
    init_stack_net,
)
from mlff.properties import property_names as pn
from mlff.training import Coach


def unit_convert_data(x: Dict, table: Dict):
    """
    Convert units in the data dictionary.

    Args:
        x (Dict): The data dictionary.
        table (Dict): A dictionary mapping quantities to conversion factors.

    Returns: The data dictionary with converted quantities.

    """
    for (k, v) in x.items():
        if k in list(table.keys()):
            print('Converted {} to ase default unit.'.format(k))
            x[k] *= table[k]
    return x


# TODO: save predictions to file
def evaluate():
    # Create the parser
    parser = argparse.ArgumentParser(description='Evaluate a NN.')

    # Add the arguments
    parser.add_argument('--ckpt_dir', type=str, required=False, default=os.getcwd(),
                        help='Path to the checkpoint directory. Defaults to the current directory.')

    parser.add_argument('--apply_to', type=str, required=False, default=None,
                        help='Path to data file that the model should be applied to. '
                             'Defaults to the training data file.')

    parser.add_argument('--on', type=str, required=False, default='test',
                        help='Evaluate the model on the `train`,`valid` or `test` split. Defaults to `test`.')

    # Arguments that determine the training parameters
    parser.add_argument('--n_test', type=int, required=False, default=None,
                        help="Number of test points. Defaults to all data points that have been not seen during "
                             "training if model is evaluated on the same data set as it has been trained on.")

    parser.add_argument('--batch_size', type=int, required=False, default=10,
                        help="Batch size of the inference passes. Default=10")

    parser.add_argument('--from_split', type=str, required=False, default=None,
                        help='The name of the data split. If not specified, all data from the file specified in '
                             '`--apply_to` is loaded and used for testing.')

    parser.add_argument("--units", action=StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...", default=None,
                        help='Units in the data set for the quantities. Needs only to be specified'
                             'if the model has been trained on units different from the ones present in the data set.')

    parser.add_argument("--prop_keys", action=StoreDictKeyPair, metavar="KEY1=VAL1,KEY2=VAL2...", default=None,
                        help='Property keys of the data set. Needs only to be specified, if e.g. the keys of the '
                             'properties in the data set that the model is applied to differ from the keys the model'
                             'has been trained on.')

    parser.add_argument('--neigh_cut', type=float, required=False, default=None,
                        help="Cutoff used for the calculation of the neighborhood lists. Defaults to the r_cut"
                             "of the NN model.")

    parser.add_argument('--targets', nargs='+', required=False, default=None)

    parser.add_argument('--jax_dtype', type=str, required=False, default='x32',
                        help='Set JAX default dtype. Default is jax.numpy.float32')

    parser.add_argument('--save_predictions_to', type=str, required=False, default='predictions.npz',
                        help='Save the predictions and ground truth values to a ckpt_dir/$save_predictions_to.npz.')

    args = parser.parse_args()

    # Read arguments
    ckpt_dir = args.ckpt_dir
    batch_size = args.batch_size
    n_test = args.n_test
    apply_to = args.apply_to
    from_split = args.from_split
    units = args.units
    prop_keys = args.prop_keys
    n_cut = args.neigh_cut
    _targets = args.targets
    save_predictions_to = args.save_predictions_to

    evaluate_on = args.on

    jax_dtype = args.jax_dtype
    if jax_dtype == 'x64':
        from jax.config import config
        config.update("jax_enable_x64", True)

    ckpt_dir = (Path(args.ckpt_dir).absolute().resolve()).as_posix()

    h = read_json(os.path.join(ckpt_dir, 'hyperparameters.json'))

    coach = Coach(**h['coach'])

    targets = _targets if _targets is not None else coach.targets

    scales = read_json(os.path.join(ckpt_dir, 'scales.json'))

    test_net = init_stack_net(h)
    _prop_keys = test_net.prop_keys
    if prop_keys is not None:
        _prop_keys.update(prop_keys)
        test_net.reset_prop_keys(prop_keys=_prop_keys)
    prop_keys = test_net.prop_keys

    conversion_table = {}
    if units is not None:
        for (q, v) in units.items():
            k = prop_keys[q]
            conversion_table[k] = eval(v)

    r_cut = [x[list(x.keys())[0]]['r_cut'] for x in h['stack_net']['geometry_embeddings'] if
             list(x.keys())[0] == 'geometry_embed'][0]
    mic = [x[list(x.keys())[0]]['mic'] for x in h['stack_net']['geometry_embeddings'] if
           list(x.keys())[0] == 'geometry_embed'][0]

    if n_cut is not None:
        r_cut = n_cut
        if n_cut < r_cut:
            logging.warning(f"The specified cutoff for neighborhood calculations n_cut={n_cut} is smaller than the "
                            f"model cutoff r_cut={r_cut}. This will likely result in wrong model prediction.")

    test_params = load_params_from_ckpt_dir(ckpt_dir)

    if pn.force in targets:
        if pn.stress in targets:
            _test_obs_fn = jax.jit(jax.vmap(get_energy_force_stress_fn(test_net), in_axes=(None, 0)))
        else:
            _test_obs_fn = jax.jit(jax.vmap(get_obs_and_force_fn(test_net), in_axes=(None, 0)))
    else:
        _test_obs_fn = jax.jit(jax.vmap(get_observable_fn(test_net), in_axes=(None, 0)))

    prop_keys_inv = {v: k for (k, v) in prop_keys.items()}

    def scale(_k, _v):
        return scales[prop_keys_inv[_k]]['scale'] * _v

    def shift(_k, _v, _z):
        shifts = np.array(scales[prop_keys_inv[_k]]['per_atom_shift'], np.float64)[_z.astype(int)].sum(axis=-1)  # shape: (B)
        return _v + np.expand_dims(shifts, [i for i in range(1, _v.ndim)])

    def scale_and_shift_fn(_x: Dict, _z: np.ndarray):
        return {_k: shift(_k, scale(_k, _v), _z) for (_k, _v) in _x.items()}

    def test_obs_fn(params, inputs):
        z_key = prop_keys[pn.atomic_type]
        nn_out = jax.tree_map(lambda _x: np.array(_x, dtype=np.float64), _test_obs_fn(params, inputs))
        z = inputs[z_key]  # shape: (B,n)

        if pn.stress in targets:
            stress_key = prop_keys[pn.stress]
            cell_key = prop_keys[pn.unit_cell]

            cell = inputs[cell_key]  # shape: (B,3,3)
            cell_volumes = np.abs(np.linalg.det(cell))  # shape: (B)
            scaled_stress = nn_out[stress_key]  # shape: (B,3,3)
            stress = scaled_stress / cell_volumes[:, None, None]  # shape: (B,3,3)
            nn_out[stress_key] = stress

        return scale_and_shift_fn(nn_out, z)

    # one can either load train, valid or test split
    if evaluate_on == 'train':
        n_train = h['dataset']['split']['n_train']
        n_valid = 0
        n_test = 0
    elif evaluate_on == 'valid':
        n_train = 0
        n_valid = h['dataset']['split']['n_valid']
        n_test = 0
    elif evaluate_on == 'test':
        n_train = 0
        n_valid = 0
        # n_test has already been set above
    else:
        raise ValueError(f'`--on {evaluate_on}` is invalid. Use one of `train`, `valid`, `test`.')

    # if apply_to and from_split are not specified default split_from to default split_name 'split' from DataSet.
    if apply_to is None and from_split is None:
        print('Loading ')
        from_split = 'split'

    if apply_to is None and from_split is not None:
        print('Loading evaluation points according to the saved {} split in {} from {}.'
              .format(evaluate_on, os.path.join(ckpt_dir, 'splits.json'), coach.data_path))

        data_path = Path(coach.data_path)

        if not data_path.is_absolute():
            data_path = (Path().absolute().parent / data_path).resolve()

        if data_path.suffix == '.npz':
            test_data = dict(np.load(data_path))
        else:
            from mlff.data import AseDataLoader
            load_stress = pn.stress in targets
            data_loader = AseDataLoader(data_path, load_stress=load_stress)
            test_data = dict(data_loader.load_all())

        # test_data = dict(np.load(data_path))
        test_data = unit_convert_data(test_data, table=conversion_table)
        test_data_set = DataSet(prop_keys=prop_keys, data=test_data)
        test_data_set.load_split(file=os.path.join(ckpt_dir, 'splits.json'),
                                 n_train=n_train,
                                 n_valid=n_valid,
                                 n_test=n_test,
                                 r_cut=r_cut,
                                 mic=mic,
                                 split_name=from_split)
        d_test = test_data_set.get_data_split()[evaluate_on]

    elif apply_to is not None and from_split is None:
        print(f'Loading test points from {apply_to}.')
        if Path(apply_to).suffix == '.npz':
            test_data = dict(np.load(apply_to))
        else:
            from mlff.data import AseDataLoader
            load_stress = pn.stress in targets
            data_loader = AseDataLoader(apply_to, load_stress=load_stress)
            test_data = dict(data_loader.load_all())

        test_data = unit_convert_data(test_data, table=conversion_table)
        test_data_set = DataSet(prop_keys=prop_keys, data=test_data)

        test_data_set.random_split(n_train=0,
                                   n_valid=0,
                                   n_test=n_test,
                                   mic=mic,
                                   training=False,
                                   r_cut=r_cut)
        d_test = test_data_set.get_data_split()['test']

    elif apply_to is not None and from_split is not None:
        print('Loading evaluation points according to the saved {} split in {} from {}.'
              .format(evaluate_on, os.path.join(ckpt_dir, 'splits.json'), apply_to))

        if Path(apply_to).suffix == '.npz':
            test_data = dict(np.load(apply_to))
        else:
            from mlff.data import AseDataLoader
            load_stress = pn.stress in targets
            data_loader = AseDataLoader(apply_to, load_stress=load_stress)
            test_data = dict(data_loader.load_all())

        test_data = unit_convert_data(test_data, table=conversion_table)
        test_data_set = DataSet(prop_keys=prop_keys, data=test_data)
        test_data_set.load_split(file=os.path.join(ckpt_dir, 'splits.json'),
                                 n_train=n_train,
                                 n_valid=n_valid,
                                 n_test=n_test,
                                 r_cut=r_cut,
                                 mic=mic,
                                 split_name=from_split)
        d_test = test_data_set.get_data_split()[evaluate_on]
    else:
        msg = 'One should not end up here. This is likely due to a bug in the mlff package. Please report to ' \
              'https://github.com/thorben-frank/mlff'
        raise RuntimeError(msg)

    test_data_tuple = DataTuple(inputs=coach.inputs,
                                targets=targets,
                                prop_keys=prop_keys)

    test_input, test_obs = test_data_tuple(d_test)

    test_metrics, test_obs_pred = evaluate_model(params=test_params,
                                                 obs_fn=test_obs_fn,
                                                 data=(test_input, test_obs),
                                                 batch_size=batch_size,
                                                 metric_fn={'mae': partial(mae_metric, pad_value=0),
                                                            'rmse': partial(rmse_metric, pad_value=0),
                                                            'R2': partial(r2_metric, pad_value=0)}
                                                 )
    print(f'Metrics on the {evaluate_on} data split: ')
    pprint(test_metrics)

    with open(os.path.join(ckpt_dir, f'metrics_on_{evaluate_on}.json'), 'w') as f:
        json.dump(test_metrics, f, indent=1)
    if save_predictions_to is not None:
        p = Path(save_predictions_to)
        _save_predictions_to = f'{p.stem}_on_{evaluate_on}{p.suffix}'
        np.savez(os.path.join(ckpt_dir, _save_predictions_to), **test_obs_pred)


if __name__ == '__main__':
    evaluate()
