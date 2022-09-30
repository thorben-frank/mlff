import numpy as np
from typing import (Any, Callable, Dict, Sequence)
import logging
logging.basicConfig(level=logging.INFO)
Array = Any


def evaluate_model(params,
                   fn: Callable,
                   inputs,
                   batch_size: int,
                   ) -> Array:
    """
    Evaluate the model on some inputs given the model functions as well as a batch size.
    Args:
        params (Dict): Parameters of the model. Is passed to fn.
        fn (Callable): Model function that maps inputs to model outputs.
        inputs (): The inputs. Either a dictionary or an npz with keys ['R', 'z']
        batch_size (int): The batch size in which the inputs are passed in the model.

    Returns: Outputs of the model function, shape: [(N_data // batch_size) * batch_size, ...]

    """
    n_data = inputs['R'].shape[0]
    n_batches = n_data // batch_size
    if n_data % batch_size != 0:
        n_left_out = n_data % batch_size
        logging.warning(
            "Number of data points modulo `batch_size` is unequal to zero. Left out {} data points at the end"
            " of the data.".format(n_left_out)
        )
    y_model = []
    for b in range(n_batches):
        logging.info("Evaluate batch {} from {}".format(b + 1, n_batches))
        idx_lower = b * batch_size
        idx_upper = (b + 1) * batch_size
        R_ = inputs['R'][idx_lower:idx_upper, ...]
        z_ = inputs['z'][idx_lower:idx_upper, ...]
        y_model += [fn(params, R_, z_)]

    y_model = np.concatenate(y_model, 0)
    return y_model


def evaluate_model_indexed(params,
                           fn: Callable,
                           inputs,
                           batch_size: int
                           ) -> Array:
    """
    Evaluate the model on some inputs given the model functions as well as a batch size.
    Args:
        params (Dict): Parameters of the model. Is passed to fn.
        fn (Callable): Model function that maps inputs to model outputs.
        inputs (): The inputs. Either a dictionary or an npz with keys ['R', 'z', 'idx_i', 'idx_j']
        batch_size (int): The batch size in which the inputs are passed in the model.

    Returns: Outputs of the model function, shape: [(N_data // batch_size) * batch_size, ...]

    """
    n_data = inputs['R'].shape[0]
    n_batches = n_data // batch_size
    if n_data % batch_size != 0:
        n_left_out = n_data % batch_size
        logging.warning(
            "Number of data points modulo `batch_size` is unequal to zero. Left out {} data points at the end"
            " of the data.".format(n_left_out)
        )
    y_model = []
    for b in range(n_batches):
        logging.info("Evaluate batch {} from {}".format(b + 1, n_batches))
        idx_lower = b * batch_size
        idx_upper = (b + 1) * batch_size
        R_ = inputs['R'][idx_lower:idx_upper, ...]
        z_ = inputs['z'][idx_lower:idx_upper, ...]
        idx_i = inputs['idx_i'][idx_lower:idx_upper, ...]
        idx_j = inputs['idx_j'][idx_lower:idx_upper, ...]
        y_model += [fn(params, R_, z_, idx_i, idx_j)]

    y_model = np.concatenate(y_model, 0)
    return y_model
