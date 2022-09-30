import optax
from flax import traverse_util
from flax.core.frozen_dict import freeze, unfreeze
from optax import constant_schedule
from dataclasses import dataclass

import logging
from typing import (Dict)
from optax import exponential_decay


@dataclass
class Optimizer:
    b1: float = 0.9
    b2: float = 0.999
    eps: float = 1e-8
    eps_root: float = 0.0
    transition_steps: int = None
    decay_rate: float = None
    weight_decay: float = None
    clip_by_global_norm: float = None

    def get(self,
            learning_rate: float,
            *args,
            **kwargs) -> optax.GradientTransformation:
        """
        Get the optax optimizer, for a specified learning rate.

        Args:
            learning_rate (float): The learning rate
            *args ():
            **kwargs ():

        Returns:

        """
        self.learning_rate = learning_rate
        weight_decay = 0. if self.weight_decay is None else self.weight_decay
        mask = None if self.weight_decay is None else flattened_traversal(lambda path, _: path[-1] != 'bias')

        if self.transition_steps is None or self.decay_rate is None:
            step_size_fn = None
        else:
            step_size_fn = exponential_decay(init_value=1.,
                                             transition_steps=self.transition_steps,
                                             decay_rate=self.decay_rate
                                             )

        return optimizer(learning_rate=self.learning_rate,
                                                   b1=self.b1,
                                                   b2=self.b2,
                                                   eps=self.eps,
                                                   eps_root=self.eps_root,
                                                   weight_decay=weight_decay,
                                                   mask=mask,
                                                   step_size_fn=step_size_fn,
                                                   clip_by_global_norm=self.clip_by_global_norm)

    def __dict_repr__(self):
        return {'optimizer': {'learning_rate': self.learning_rate,
                              'transition_steps': self.transition_steps,
                              'decay_rate': self.decay_rate,
                              'weight_decay': self.weight_decay,
                              'clip_by_global_norm': self.clip_by_global_norm}}


def optimizer(learning_rate,
              b1: float = 0.9,
              b2: float = 0.999,
              eps: float = 1e-8,
              eps_root: float = 0.0,
              weight_decay: float = 0.0,
              mask=None,
              step_size_fn=None,
              clip_by_global_norm=None):

    if clip_by_global_norm is None:
        clip_fn = optax.scale(1.)
    else:
        clip_fn = optax.clip_by_global_norm(max_norm=clip_by_global_norm)

    if step_size_fn is not None:
        msg = 'Scheduled LR decay has been moved to the train_state class. No LR decay is used.'
        raise DeprecationWarning(msg)

    # if step_size_fn is None:
    #     step_size_fn = constant_schedule(1.)

    return optax.chain(
        clip_fn,
        optax.scale_by_adam(b1=b1, b2=b2, eps=eps, eps_root=eps_root),
        optax.add_decayed_weights(weight_decay, mask),
        optax.inject_hyperparams(optax.scale)(step_size=-learning_rate),
        # optax.scale_by_schedule(step_size_fn),
    )


def flattened_traversal(fn):
    def mask(data):
        flat = traverse_util.flatten_dict(unfreeze(data))
        return freeze(traverse_util.unflatten_dict({k: fn(k, v) for k, v in flat.items()}))
    return mask


def decay_mask_fn(params):
    flat_params = traverse_util.flatten_dict(unfreeze(params))
    flat_mask = {path: (path[-1] != "bias" and path[-2:] != ("LayerNorm", "scale")) for path in flat_params}
    return freeze(traverse_util.unflatten_dict(flat_mask))


# TODO: legacy, take care of secure removal

#                                                                                                                      #
#                                     Initialize optimizer from dictionary/json                                        #
#                                                                                                                      #


OPTIMIZER_hyperparameters = {'learning_rate': None,
                             'weight_decay': None,
                             'transition_steps': None,
                             'decay_rate': None
                             }


def hyper_params_to_properties(hyper_params):
    lr = hyper_params['learning_rate']
    if lr is None:
        raise ValueError("Learning rate must be specified in the optimizer hyperparameters"
                         "since no default value has been set")

    weight_decay = 0. if hyper_params['weight_decay'] is None else hyper_params['weight_decay']
    mask = None if hyper_params['weight_decay'] is None else flattened_traversal(lambda path, _: path[-1] != 'bias')
    if hyper_params['transition_steps'] is None or hyper_params['decay_rate'] is None:
        step_size_fn = None
    else:
        step_size_fn = exponential_decay(1.,
                                         transition_steps=hyper_params['transition_steps'],
                                         decay_rate=hyper_params['decay_rate']
                                         )

    return {'learning_rate': lr,
            'weight_decay': weight_decay,
            'mask': mask,
            'step_size_fn': step_size_fn
            }


def optimizer_from_hyper_params(hyper_params: Dict):
    d = {}
    for k, v_default in OPTIMIZER_hyperparameters.items():
        try:
            v = hyper_params[k]
        except KeyError:
            v = v_default
            logging.warning('The argument {} is missing in the optimizer hyperparameters. Set default '
                            '{}={}.'.format(k, k, v_default))

        d[k] = v

    d = hyper_params_to_properties(d)
    return optimizer(**d)


# class ReduceLrOnPlateauState(NamedTuple):
#     plateau_count: check.Array
#     plateau_length: check.Array
#     loss: check.Array
#
#
#   def update_fn(updates, state, params=None):
#       del params
#       mu = _update_moment(updates, state.mu, b1, 1)
#       nu = _update_moment_per_elem_norm(updates, state.nu, b2, 2)
#       count_inc = numerics.safe_int32_increment(state.count)
#       mu_hat = _bias_correction(mu, b1, count_inc)
#       nu_hat = _bias_correction(nu, b2, count_inc)
#       updates = jax.tree_map(
#           lambda m, v: m / (jnp.sqrt(v + eps_root) + eps), mu_hat, nu_hat)
#       mu = utils.cast_tree(mu, mu_dtype)
#       return updates, ScaleByAdamState(count=count_inc, mu=mu, nu=nu)
#
#
# class ScaleByAdamState(NamedTuple):
#   """State for the Adam algorithm."""
#   count: chex.Array  # shape=(), dtype=jnp.int32.
#   mu: base.Updates
#   nu: base.Updates
#
#
# def scale_by_adam(
#     b1: float = 0.9,
#     b2: float = 0.999,
#     eps: float = 1e-8,
#     eps_root: float = 0.0,
#     mu_dtype: Optional[Any] = None,
# ) -> base.GradientTransformation:
#   """Rescale updates according to the Adam algorithm.
#
#   References:
#     [Kingma et al, 2014](https://arxiv.org/abs/1412.6980)
#
#   Args:
#     b1: decay rate for the exponentially weighted average of grads.
#     b2: decay rate for the exponentially weighted average of squared grads.
#     eps: term added to the denominator to improve numerical stability.
#     eps_root: term added to the denominator inside the square-root to improve
#       numerical stability when backpropagating gradients through the rescaling.
#     mu_dtype: optional `dtype` to be used for the first order accumulator; if
#       `None` then the `dtype is inferred from `params` and `updates`.
#
#   Returns:
#     An (init_fn, update_fn) tuple.
#   """
#
#   mu_dtype = utils.canonicalize_dtype(mu_dtype)
#
#   def init_fn(params):
#     mu = jax.tree_map(  # First moment
#         lambda t: jnp.zeros_like(t, dtype=mu_dtype), params)
#     nu = jax.tree_map(jnp.zeros_like, params)  # Second moment
#     return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)
#
#   def update_fn(updates, state, params=None):
#     del params
#     mu = _update_moment(updates, state.mu, b1, 1)
#     nu = _update_moment_per_elem_norm(updates, state.nu, b2, 2)
#     count_inc = numerics.safe_int32_increment(state.count)
#     mu_hat = _bias_correction(mu, b1, count_inc)
#     nu_hat = _bias_correction(nu, b2, count_inc)
#     updates = jax.tree_map(
#         lambda m, v: m / (jnp.sqrt(v + eps_root) + eps), mu_hat, nu_hat)
#     mu = utils.cast_tree(mu, mu_dtype)
#     return updates, ScaleByAdamState(count=count_inc, mu=mu, nu=nu)
#   return base.GradientTransformation(init_fn, update_fn)

from typing import NamedTuple


def update_state_with_loss(opt_state, loss_scalar):
    pass


# class ReduceLrOnPlateauState(NamedTuple):
#     plateau_count: check.Array
#     plateau_length: check.Array
#     loss_old: check.Array
#     loss_new: check.Array
#
#
# def reduce_lr_on_plateau(patience, decay_factor):
#     def update_fn(self, updates, state, params=None):
#         del params
#         improved = state.loss_new // state.loss_old  # 0 if loss_new < loss_old
#         plateau_length = improved * (state.plateau_length + improved)
#         _plateau_length = plateau_length % patience
#         _plateau_count = self.plateau_length // patience
#         return updates, ReduceLrOnPlateauState()