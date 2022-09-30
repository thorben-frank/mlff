import os
import pathlib
import json
from typing import (Dict, Sequence)


def read_json(path):
    with open(path) as json_file:
        data = json.load(json_file)
    return data


def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + "_" + str(counter) + extension
        counter += 1

    return path


def create_directory(path, exists_ok=False):
    if not exists_ok:
        path = uniquify(path)
    pathlib.Path(path).mkdir(parents=True, exist_ok=exists_ok)
    return path


def save_dict(path, filename, data, exists_ok=False):
    path = create_directory(path, exists_ok=exists_ok)
    save_path = os.path.join(path, filename)
    with open(save_path, 'w') as f:
        json.dump(data, f)


def bundle_dicts(x: Sequence[Dict]) -> Dict:
    """
    Bundles a list of dictionaries into one.

    Args:
        x (Sequence): List of dictionaries.

    Returns: The bundled dictionary.

    """

    bd = {}
    for d in x:
        bd.update(d)
    return bd


def merge_dicts(x, y):
    x.update(y)
    return x
