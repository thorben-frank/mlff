import jax
import yaml
import argparse
import logging
import copy
from functools import partial

from typing import (Any, Dict, Tuple)

from mlff.src.training import Coach
from mlff.src.nn.stacknet import StackNet, init_stack_net
from mlff.src.training.train_state import CustomTrainState


def _parse_yaml(config_file: str):
    """
    Parse a YAML config file to a dictionary.

    Args:
        config_file (str): Path to the YAML config file.

    Returns: The YAML file as dictionary.

    """
    with open(config_file, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
            return parsed_yaml
        except yaml.YAMLError as exc:
            logging.error(exc)


def recursive_items(dictionary: Dict) -> Tuple:
    for key, value in dictionary.items():
        if type(value) is dict:
            yield from recursive_items(value)
            yield key, value
        elif type(value) is list:
            for x in value:
                if type(x) is dict:
                    yield from recursive_items(x)
                else:
                    yield key, value
        else:
            yield key, value


def make_hash(o: Dict) -> Any:
    """
    Makes a hash from a dictionary, list, tuple or set to any level, that contains
    only other hashable types (including any lists, tuples, sets, and
    dictionaries).
    """

    if isinstance(o, (set, tuple, list)):

        return tuple([make_hash(e) for e in o])

    elif not isinstance(o, dict):

        return hash(o)

    new_o = copy.deepcopy(o)
    for k, v in new_o.items():
        new_o[k] = make_hash(v)

    return hash(tuple(frozenset(sorted(new_o.items()))))


def replace_fn(x: Any, key2value_map: Dict) -> Any:
    if str(x)[0] == '$':
        return key2value_map[x[1:]]
    else:
        return x


def read_yaml_config_file(config_file: str) -> Dict:
    """
    Read a YAML config file and return a fully build hyperparameter dictionary which can be used to initialize a
    StackNet, Coach or TrainState. Function takes care of referencing syntax that can be used to get rid of redundant
    variables.

    Args:
        config_file (str): Path to the config file.

    Returns: Dictionary of hyperparameters.

    """
    _h = _parse_yaml(config_file=config_file)

    _keys = []
    key2value = {}
    for (k, v) in recursive_items(_h):
        if k in _keys:
            pass
        else:
            _keys += [k]
            key2value[k] = v

    # Iterate over the keys until self-consistency is reached. This corresponds to replacing all $references
    # with the corresponding value. As the references can be nested, a single replace pass might be not enough.
    # Instead each dictionary is hashed and compared to the dictionary before until no changes are observed anymore.
    _d_hash = 0
    d_hash = 1
    while _d_hash != d_hash:
        _d_hash = make_hash(key2value)
        key2value = jax.tree_map(partial(replace_fn, key2value_map=key2value), key2value)
        d_hash = make_hash(key2value)

    h = jax.tree_map(partial(replace_fn, key2value_map=key2value), _h)

    return h


def _fetch_coach_config(config: Dict) -> Dict:
    return config['coach']


def _fetch_config(config: Dict, key: str) -> Dict:
    return config[key]


def _fetch_stack_net_config(config: Dict) -> Dict:
    return config['stack_net']


def _create_coach(config: Dict) -> Coach:
    return Coach(**_fetch_coach_config(config=config))


def _create_stack_net(config: Dict) -> StackNet:
    return init_stack_net(config)
