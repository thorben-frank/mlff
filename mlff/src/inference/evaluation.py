import jax
import jax.numpy as jnp
import numpy as np
import logging

from tqdm import tqdm
from typing import (Any, Callable, Dict, Tuple, Union)
logging.basicConfig(level=logging.INFO)
Array = Any


def tree_concatenate_tensors(py_tree_1, py_tree_2, axis=0):
    return jax.tree_map(lambda x, y: np.concatenate([x, y], axis=axis), py_tree_1, py_tree_2)


def evaluate_model(params,
                   obs_fn: Callable,
                   data: Tuple[Dict, Union[Dict, None]],
                   batch_size: int,
                   metric_fn: Dict[str, Callable] = None,
                   verbose=True) -> Tuple[Dict, Dict]:
    """
    Evaluate a model, given its params `params` and an observable function `obs_fn`. One can either pass a tuple
    to `data` where the first entry is the input to the observable function and the second is the expected output, or
    a tuple where the second entry is `None` if target data is not known (needed). The function returns the metrics, as
    well as the inputs, predictions and targets that have been used. Since the number of input data modulo the batch
    size can be unequal to zero, the returned quantities can have smaller data dimension than the original data.

    Args:
        params (FrozenDict): FLAX model parameters.
        obs_fn (Callable): Observable function.
        data (DataTuple): A tuple of data, where the first is the input to the observable function and the latter
            the expected output. If the output is not known, pass a Tuple with empty second argument instead.
        batch_size (int): Batch size used to split the data.
        metric_fn (Dict): Dictionary with `metric_name` as key and a Callable as value that represents a metric. It
            should be noted that the metric_fn must take care of potentially necessary reshaping of the tensors or
            excluding padded values. Standard metrics that do that lie under `from mlff.src.inference.evaluation`
        verbose (bool): Log additional information.

    Returns: Tuple of Dictionaries. The first contains the metrics and the second the inputs, predictions and targets.

    """

    inputs, targets = data

    _k = list(inputs.keys())[0]
    n_data = len(inputs[_k])
    n_batches = n_data // batch_size

    if n_data % batch_size != 0:
        n_left_out = n_data % batch_size
        logging.warning(
            "Number of data points modulo `batch_size` is unequal to zero. Left out {} data points at the end"
            " of the data.".format(n_left_out)
        )

    idxs = jnp.arange(n_data)
    idxs = idxs[:n_batches * batch_size]  # skip incomplete batch
    idxs = idxs.reshape((n_batches, batch_size))

    obs_pred = {}
    logging.info('Model evaluation (Total number of data: {}, number of batches: {}, batch size: {}'
                 .format(int(n_batches*batch_size), n_batches, batch_size))
    for (i, idx) in enumerate(tqdm(idxs)):
        # if verbose:
        #     logging.info("Evaluate batch {} from {}".format(i + 1, n_batches))

        input_batch = jax.tree_map(lambda y: y[idx, ...], inputs)
        obs_pred_ = obs_fn(params, input_batch)
        if len(obs_pred) == 0:
            obs_pred.update(obs_pred_)
        else:
            obs_pred = tree_concatenate_tensors(obs_pred, obs_pred_, axis=0)

    inputs = jax.tree_map(lambda x: x[:int(n_batches * batch_size)], inputs)
    if targets is not None:
        targets = jax.tree_map(lambda x: x[:int(n_batches * batch_size)], targets)

    # make sure that metrics are only calculated for quantities that appear in both, the output and the target.
    if len(set(obs_pred.keys()) ^ set(targets.keys())) != 0:
        _joint_keys = set(obs_pred.keys()) & set(targets.keys())
        logging.warning('The model predicts quantities with keys {} whereas the data it is '
                        'evaluated on has keys {}. Only evaluating the model on joint keys {}.'
                        .format(list(obs_pred.keys()), list(targets.keys()), _joint_keys))
        obs_pred_eval = {k: v for (k, v) in obs_pred.items() if k in _joint_keys}
        target_eval = {k: v for (k, v) in targets.items() if k in _joint_keys}
    else:
        obs_pred_eval = obs_pred
        target_eval = targets

    metrics = {}
    if metric_fn is not None:
        # flattened_predictions = jax.tree_map(lambda x: x[:int(n_batches * batch_size)], obs_pred)
        for m_name, m_fn in metric_fn.items():
            metrics[m_name] = jax.tree_map(lambda x, y: m_fn(prediction=x, target=y), obs_pred_eval, target_eval)

    return metrics, {'inputs': inputs, 'predictions': obs_pred, 'targets': targets}


def mae_metric(prediction: Array, target: Array, pad_value: float = None) -> Array:
    """
    Metric function for mean absolute error. Padding values can be excluded from the metric calculation.

    Args:
        prediction (Array): The predictions, shape: (...)
        target (Array): The targets, shape: (...)
        pad_value (float): Value of padded values.

    Returns: scalar value, shape: (1)

    """
    p = prediction.reshape(-1)
    t = target.reshape(-1)

    if pad_value is not None:
        not_pad_idx = t != pad_value
    else:
        not_pad_idx = np.arange(len(p))

    return np.abs(p[not_pad_idx] - t[not_pad_idx]).mean()


def mse_metric(prediction: Array, target: Array, pad_value: float = None) -> Array:
    """
    Metric function for mean squared error. Padding values can be excluded from the metric calculation.

    Args:
        prediction (Array): The predictions, shape: (...)
        target (Array): The targets, shape: (...)
        pad_value (float): Value of padded values.

    Returns: scalar value, shape: (1)

    """
    p = prediction.reshape(-1)
    t = target.reshape(-1)

    if pad_value is not None:
        not_pad_idx = t != pad_value
    else:
        not_pad_idx = np.arange(len(p))

    return ((p[not_pad_idx] - t[not_pad_idx])**2).mean()


def rmse_metric(prediction: Array, target: Array, pad_value: float = None) -> Array:
    """
    Metric function for root mean square error. Padding values can be excluded from the metric calculation.

    Args:
        prediction (Array): The predictions, shape: (...)
        target (Array): The targets, shape: (...)
        pad_value (float): Value of padded values.

    Returns: scalar value, shape: (1)

    """
    return np.sqrt(mse_metric(prediction=prediction, target=target, pad_value=pad_value))
