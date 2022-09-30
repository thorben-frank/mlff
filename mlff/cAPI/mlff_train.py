import argparse
import jax
import jax.numpy as jnp
import numpy as np
import wandb
import logging
import os
import os.path

# this import is actually needed though not detected by pyCharm since it is hidden in the eval() statement
from ase.units import *
from typing import (Any, Dict)

from mlff.cAPI.process_yaml import read_yaml_config_file, _create_coach, _create_stack_net, _fetch_config
from mlff.src.training.train_state import create_train_state
from mlff.src.training import Optimizer, get_loss_fn
from mlff.src.data import DataSet, DataTuple
from mlff.src.nn.stacknet import StackNet, get_observable_fn, get_obs_and_force_fn
from mlff.src.io import bundle_dicts, save_dict, create_directory
from mlff.src.nn.embed import get_embedding_module
from mlff.src.nn.observable import get_observable_module
from mlff.src.nn.layer import get_layer
from mlff.cAPI.process_argparse import default_access


def dict_to_list(x: Dict) -> list:
    """
    Convert a dictionary of the form {ind: value} to a list where position and value of the i-th entry of the list
    corresponds to the key and value of the dictionary. Intermediate, non appearing indices are filled with zero.

    Args:
        x (dict): Dictionary of the form {ind: value}

    Returns: A list filled with the values from the dictionary.

    """

    k = list(x.keys())
    v = list(x.values())
    arr = np.zeros(max(k))
    np.put(arr, ind=np.array(k, dtype=int), v=np.array(v, dtype=np.float64))
    return arr.tolist()


def train():
    # Create the parser
    parser = argparse.ArgumentParser(description='Run a NN training from YAML file.')

    parser.add_argument('--config_file', type=str, required=False, default=None,
                        help='Path to the YAML config file. By default checks the current directory for any YAML '
                             'files and runs the first it finds.')

    args = parser.parse_args()
    config_file = args.config_file

    if config_file is None:
        auto_detect_yaml = False
        for entry in os.scandir(os.getcwd()):
            extension = os.path.splitext(entry.name)[1]
            if extension.lower() == '.yaml':
                auto_detect_yaml = True
                config_file = entry.path
                break

        if auto_detect_yaml is False:
            msg = "Did not find a YAML file in the current directory {}. Either switch to a directory with a YAML " \
                  "file or specify the location using the `--config_file` argument.".format(os.getcwd())
            raise RuntimeError(msg)

    # load the config file
    logging.info("Load training specifications from {}.".format(config_file))
    config = read_yaml_config_file(config_file=config_file)

    # read prop_keys
    prop_keys = _fetch_config(config=config, key='prop_keys')
    inv_prop_keys = {v: k for (k, v) in prop_keys.items()}
    E_key = default_access(prop_keys, key='energy', default=None)
    F_key = default_access(prop_keys, key='force', default=None)
    z_key = default_access(prop_keys, key='atomic_type', default=None)

    # read units
    units: Dict[str, str] = config['units']
    conversion_table = {}
    for (k, v) in units.items():
        q = k.split('_')[0]
        _k = prop_keys[q]
        conversion_table[_k] = eval(v)

    # coach properties
    coach = _create_coach(config=config)
    _ckpt_dir = create_directory(coach.ckpt_dir, exists_ok=False)
    if _ckpt_dir != coach.ckpt_dir:
        logging.warning('The specified checkpoint directory ckpt_dir={} already exists. Changing it to {}'
                        .format(coach.ckpt_dir, _ckpt_dir)
                        )
        coach.ckpt_dir = _ckpt_dir

    coach_config = config['coach']
    loss_weights: Dict[str, float] = coach_config['loss_weights']
    _loss_weights = {}
    for (k, v) in loss_weights.items():
        q = k.split('_')[0]
        _k = prop_keys[q]
        _loss_weights[_k] = v

    coach.loss_weights = _loss_weights

    # run training config
    run_config = _fetch_config(config=config, key='run')

    # train state config
    train_state_config = _fetch_config(config=config, key='train_state')
    learning_rate = train_state_config['learning_rate']
    clip_by_global_norm = default_access(train_state_config, key='clip_by_global_norm', default=None)
    reduce_lr_on_plateau = default_access(train_state_config, key='reduce_lr_on_plateau', default=None)
    polyak_step_size = default_access(train_state_config, key='polyak_step_size', default=None)
    # TODO: include constant scheduled lr decay

    opt = Optimizer(clip_by_global_norm=clip_by_global_norm)  # initialize optimizer
    tx = opt.get(learning_rate=learning_rate)  # get optax gradient transformation

    # data set config
    dataset_config = _fetch_config(config=config, key='dataset')
    r_cut_dataset = dataset_config['r_cut']
    strat_key = default_access(dataset_config, key='strat_key', default=None)
    n_test = default_access(dataset_config, key='n_test', default=None)

    data = dict(np.load(coach.data_path))
    for (k, v) in data.items():
        if k in list(conversion_table.keys()):
            logging.info('Converted {} to ase default unit.'.format(k))
            data[k] *= conversion_table[k]

    dataset = DataSet(data=data, prop_keys=prop_keys)

    if strat_key is not None:
        d = dataset.strat_split(**dataset_config, training=True, n_test=n_test)
    else:
        d = dataset.random_split(**dataset_config, training=True, n_test=n_test)

    dataset.save_splits_to_file(coach.ckpt_dir, 'split.npz')

    # net properties
    net_config = _fetch_config(config=config, key='stack_net')

    # init geometry embedding modules
    geometric_embedding_modules = []
    for x in net_config['geometry_embeddings']:
        geometric_embedding_modules += [get_embedding_module(*tuple(x.items())[0])]

    # init feature embedding modules
    feature_embedding_modules = []
    for x in net_config['feature_embeddings']:
        feature_embedding_modules += [get_embedding_module(*tuple(x.items())[0])]

    # init layers
    layers = []
    for x in net_config['layers']:
        layers += [get_layer(*tuple(x.items())[0])]

    # init observable modules
    observable_modules = []
    for x in net_config['observables']:
        module_name, module_attributes = tuple(x.items())[0]
        if module_name == 'energy':
            _per_atom_shift = default_access(module_attributes, key='per_atom_shift', default=None)
            _per_atom_scale = default_access(module_attributes, key='per_atom_scale', default=None)
            if type(_per_atom_shift) is dict:
                per_atom_shift = dict_to_list(_per_atom_shift)
            elif type(_per_atom_shift) is float:
                per_atom_shift = [_per_atom_shift] * 100
            elif _per_atom_shift is None:
                per_atom_shift = [d['train']['{}_mean'.format(E_key)] / len(d['train'][z_key][0])] * 100
            else:
                msg = 'Invalid per_atom_shift in YAML file. Valid are dictionaries, floats or null.'
                raise ValueError(msg)
            if type(_per_atom_scale) is dict:
                per_atom_scale = dict_to_list(_per_atom_scale)
            elif type(_per_atom_scale) is float:
                per_atom_scale = [_per_atom_scale] * 100
            elif _per_atom_scale is None:
                if F_key in coach.target_keys:
                    per_atom_scale = [d['train']['{}_scale'.format(F_key)]] * 100
                else:
                    per_atom_scale = [d['train']['{}_scale'.format(E_key)]] * 100
            else:
                msg = 'Invalid per_atom_scale in YAML file. Valid are dictionaries, floats or null.'
                raise ValueError(msg)
            module_attributes['per_atom_shift'] = per_atom_shift
            module_attributes['per_atom_scale'] = per_atom_scale
        observable_modules += [get_observable_module(module_name, module_attributes)]

    # check that r_cut of net is not larger than the cutoff for the data set
    r_cut_net = \
    [x[list(x.keys())[0]]['r_cut'] for x in net_config['geometry_embeddings'] if list(x.keys())[0] == 'geometry_embed'][
        0]
    if r_cut_dataset < r_cut_net:
        msg = 'The chosen cutoff radius for the dataset is smaller than the internal cutoff radius of the stack net in ' \
              'the `geometry_embed` module. '
        raise ValueError(msg)

    # init the full stacknet
    net = StackNet(geometry_embeddings=geometric_embedding_modules,
                   feature_embeddings=feature_embedding_modules,
                   layers=layers,
                   observables=observable_modules,
                   prop_keys=prop_keys)

    # wandb config
    wandb_config = _fetch_config(config=config, key='wandb')
    if 'dir' in list(wandb_config.keys()):
        pass
    else:
        wandb_config['dir'] = coach.ckpt_dir

    if prop_keys['force'] in coach.target_keys:
        obs_fn = get_obs_and_force_fn(net)
    else:
        obs_fn = get_observable_fn(net, observable_key=None)

    obs_fn = jax.vmap(obs_fn, in_axes=(None, 0))
    loss_fn = get_loss_fn(obs_fn=obs_fn, weights=coach.loss_weights)

    data_tuple = DataTuple(input_keys=coach.input_keys + ['idx_i', 'idx_j'],
                           target_keys=coach.target_keys)

    train_ds = data_tuple(d['train'])
    valid_ds = data_tuple(d['valid'])

    inputs = jax.tree_map(lambda ary: jnp.array(ary[0, ...]), train_ds[0])
    params = net.init(jax.random.PRNGKey(coach.net_seed), inputs)

    # TODO: include constant scheduled lr decay
    train_state, h_train_state = create_train_state(net,
                                                    params,
                                                    tx,
                                                    polyak_step_size=polyak_step_size,
                                                    plateau_lr_decay=reduce_lr_on_plateau)

    h_net = net.__dict_repr__()
    h_opt = opt.__dict_repr__()
    h_coach = coach.__dict_repr__()
    h_dataset = dataset.__dict_repr__()
    h = bundle_dicts([h_net, h_opt, h_coach, h_dataset, h_train_state])
    save_dict(path=coach.ckpt_dir, filename='hyperparameters.json', data=h, exists_ok=True)

    wandb.init(**wandb_config, config=h)
    coach.run(train_state=train_state,
              train_ds=train_ds,
              valid_ds=valid_ds,
              loss_fn=loss_fn,
              **run_config)


if __name__ == '__main__':
    train()
