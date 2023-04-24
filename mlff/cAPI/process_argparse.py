import argparse

from typing import (Any, Dict)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class StoreDictKeyPair(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        for kv in values.split(","):
            k, v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)


def default_access(x: Dict, key: str, default: Any) -> Any:
    """
    Access the value for some `key` of a dictionary `x`. If `key` not exists return the `default` value.

    Args:
        x (Dict): The dictionary.
        key (str): The key.
        default (Any): The default value.

    Returns: Value stored at `key` if `key` exists. Otherwise `default`.
    """

    try:
        value = x[key]
    except KeyError:
        value = default
    return value
