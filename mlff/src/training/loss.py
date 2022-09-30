import jax.numpy as jnp
from typing import (Dict, Tuple)

DataTupleT = Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]
BatchImportanceWeights = Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray], jnp.ndarray]


def mse_loss(*, y, y_true): return jnp.mean((y_true - y)**2)


def energy_loss(*, y, y_true):
    """

    Args:
        y ():
        y_true ():

    Returns:

    """
    return mse_loss(y=y, y_true=y_true)


def force_loss(*, y, y_true):
    """
    Force loss that takes differently sized molecules across a single batch into account.

    Args:
        y_pred (): predicted forces of shape [...,n,3]
        y_true (): expected forces of shape [...,n,3]

    Returns:

    """

    # number of force components per batch dimension
    n_fc = (y_true != 0).sum(-1).sum(-1)  # shape: (...)
    return jnp.mean(1/n_fc*((y - y_true)**2).sum(-1).sum(-1))


def get_loss_fn(obs_fn, weights):
    def loss_fn(params, batch: DataTupleT):
        inputs, targets = batch
        outputs = obs_fn(params, inputs)
        loss = jnp.zeros(1)
        train_metrics = {}
        for name, target in targets.items():
            _l = mse_loss(y_true=outputs[name], y=targets[name])
            loss += weights[name] * _l
            train_metrics.update({name: _l})

        # for name, w in weights.items():
        #     _l = mse_loss(y_true=outputs[name], y=targets[name])
        #     loss += w * _l
        #     train_metrics.update({name: _l})
        loss = jnp.reshape(loss, ())
        train_metrics.update({'loss': loss})

        return loss, train_metrics
    return loss_fn


def get_active_learning_loss_fn(obs_fn, weights):
    def loss_fn(params, batch: BatchImportanceWeights):
        inputs, targets, imp_weights = batch  # shape: (B, ...), (B, ...), (B)
        imp_weights_normalized = jnp.sqrt(imp_weights / imp_weights.mean())  # shape: (B)

        outputs = obs_fn(params, inputs)
        loss = jnp.zeros(1)
        train_metrics = {}
        for name, target in targets.items():
            _y = jnp.einsum('b, b... -> b...', imp_weights_normalized, outputs[name])
            _y_true = jnp.einsum('b, b... -> b...', imp_weights_normalized, targets[name])
            _l = mse_loss(y=_y, y_true=_y_true)
            loss += weights[name] * _l
            train_metrics.update({name: _l})
        # for name, w in weights.items():
        #     _l = mse_loss(y_true=outputs[name], y=targets[name])
        #     loss += w * _l
        #     train_metrics.update({name: _l})
        loss = jnp.reshape(loss, ())
        train_metrics.update({'loss': loss})

        return loss, train_metrics
    return loss_fn
