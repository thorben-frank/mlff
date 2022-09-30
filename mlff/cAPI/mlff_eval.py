import argparse

import numpy as np
import jax
import logging
import os

from ase.units import *
from pprint import pprint
from flax.training import checkpoints
from typing import (Dict)
from functools import partial

from mlff.src.training import Coach
from mlff.src.data import DataTuple, DataSet
from mlff.src.nn.stacknet import get_obs_and_force_fn, get_observable_fn
from mlff.src.io.io import read_json
from mlff.src.nn.stacknet import init_stack_net
from mlff.src.inference.evaluation import evaluate_model, mae_metric, rmse_metric
from mlff.cAPI.process_argparse import StoreDictKeyPair

logging.basicConfig(level=logging.INFO)


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
            logging.info('Converted {} to ase default unit.'.format(k))
            x[k] *= table[k]
    return x


def evaluate():

    # Create the parser
    parser = argparse.ArgumentParser(description='Evaluate a NN.')

    # Add the arguments
    parser.add_argument('--ckpt_dir', type=str, required=False, default=os.getcwd(),
                        help='Path to the checkpoint directory. Defaults to the current directory.')

    parser.add_argument('--apply_to', type=str, required=False, default=None,
                        help='Path to data file that the model should be applied to. '
                             'Defaults to the training data file.')

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

    parser.add_argument('--x64', type=bool, required=False, default=False)

    # TODO: add which quantities should be evaluated - defaults to target keys from coach

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
    x64 = args.x64

    if x64:
        from jax.config import config
        config.update("jax_enable_x64", True)

    h = read_json(os.path.join(ckpt_dir, 'hyperparameters.json'))

    coach = Coach(**h['coach'])

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

    r_cut = [x[list(x.keys())[0]]['r_cut'] for x in h['stack_net']['geometry_embeddings'] if list(x.keys())[0] == 'geometry_embed'][0]

    if n_cut is not None:
        r_cut = n_cut
        if n_cut < r_cut:
            logging.warning("The specified cutoff for neighborhood calculations n_cut={} is smaller than the model"
                            " cutoff r_cut={}. This will likely result in wrong model prediction.")

    test_params = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=None, prefix='checkpoint_loss')['valid_params']

    # TODO: better handling here, since force might not be in target keys but still could be used for prediction
    if prop_keys['force'] in coach.target_keys:
        test_obs_fn = jax.jit(jax.vmap(get_obs_and_force_fn(test_net), in_axes=(None, 0)))
    else:
        test_obs_fn = jax.jit(jax.vmap(get_observable_fn(test_net), in_axes=(None, 0)))

    if apply_to is None and from_split is None:
        msg = 'Either `apply_to` or `from_split` must be set to load the test data. Specifying both is also possible.'
        raise ValueError(msg)
    elif apply_to is None and from_split is not None:
        logging.info('Loading test points according to the saved splits in {} from {}.'
                     .format(os.path.join(ckpt_dir, 'splits.json'), coach.data_path))

        test_data = dict(np.load(coach.data_path))
        test_data = unit_convert_data(test_data, table=conversion_table)
        test_data_set = DataSet(prop_keys=prop_keys, data=test_data)
        d_test = test_data_set.load_split(file=os.path.join(ckpt_dir, 'splits.json'),
                                          n_train=0,
                                          n_valid=0,
                                          n_test=n_test,
                                          r_cut=r_cut,
                                          split_name=from_split)['test']
    elif apply_to is not None and from_split is None:
        logging.info('Loading test points from {}.'.format(apply_to))
        test_data = dict(np.load(apply_to))
        test_data = unit_convert_data(test_data, table=conversion_table)
        test_data_set = DataSet(prop_keys=prop_keys, data=test_data)

        d_test = test_data_set.random_split(n_train=0,
                                            n_valid=0,
                                            n_test=n_test,
                                            training=False,
                                            r_cut=r_cut)['test']
    elif apply_to is not None and from_split is not None:
        logging.info('Loading test points according to the saved splits in {} from {}.'
                     .format(os.path.join(ckpt_dir, 'splits.json'), apply_to))

        test_data = dict(np.load(apply_to))
        test_data = unit_convert_data(test_data, table=conversion_table)
        test_data_set = DataSet(prop_keys=prop_keys, data=test_data)
        d_test = test_data_set.load_split(file=os.path.join(ckpt_dir, 'splits.json'),
                                          n_train=0,
                                          n_valid=0,
                                          n_test=n_test,
                                          r_cut=r_cut,
                                          split_name=from_split)['test']
    else:
        msg = 'One should not end up here. This is likely due to a bug in the mlff package. Please report to ' \
              'https://github.com/thorben-frank/mlff'
        raise RuntimeError(msg)

    test_data_tuple = DataTuple(input_keys=coach.input_keys,
                                target_keys=coach.target_keys)
    test_input, test_obs = test_data_tuple(d_test)

    test_metrics, test_obs_pred = evaluate_model(params=test_params,
                                                 obs_fn=test_obs_fn,
                                                 data=(test_input, test_obs),
                                                 batch_size=batch_size,
                                                 metric_fn={'mae': partial(mae_metric, pad_value=0),
                                                            'rmse': partial(rmse_metric, pad_value=0)}
                                                 )
    pprint(test_metrics)


if __name__ == '__main__':
    evaluate()
