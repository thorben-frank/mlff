import os
import pathlib
import json
from typing import (Dict, Sequence)


def read_json(path: str) -> Dict:
    """
    Read a JSON file from path.

    Args:
        path (str): Path to JSON file.

    Returns: JSON file as dictionary.

    """
    with open(path) as json_file:
        data = json.load(json_file)
    return data


def _uniquify(path):
    """
    Uniquify `path`, i.e. if `path` exists try to create `path_1`. If this exist try to create `path_2` and so on ...

    Args:
        path (str): Directory.

    Returns: The newly created path.

    """
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + "_" + str(counter) + extension
        counter += 1

    return path


def create_directory(path, exists_ok=False) -> str:
    """
    Create the directory `path`.

    Args:
        path (str): Directory that should be created.
        exists_ok (bool): Whether the specified directory is allowed to already exist. If `False`, and the specified
        directory already exist a new directory path_1 is created. If path_1 already exists, a directory with name
        path_2 is created and so on ...

    Returns: The created path.

    """
    if not exists_ok:
        path = _uniquify(path)
    pathlib.Path(path).mkdir(parents=True, exist_ok=exists_ok)
    return path


def save_dict(path, filename, data, exists_ok=False) -> None:
    """
    Save dictionary to JSON file.

    Args:
        path (str): Directory where to save the dictionary.
        filename (str): Filename of the JSON file
        data (dict): Dictionary to save.
        exists_ok (bool): Whether the specified directory is allowed to already exist. If `False`, and the specified
        directory already exist a new directory path_1 is created. If path_1 already exists, a directory with name
        path_2 is created and so on ...

    Returns: None

    """
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
