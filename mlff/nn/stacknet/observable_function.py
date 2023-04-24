import jax.numpy as jnp
import jax
import logging

from typing import (Any, Callable, Dict, Sequence, Tuple)
from flax.core.frozen_dict import FrozenDict

logging.basicConfig(level=logging.INFO)

Array = Any
StackNet = Any
LossFn = Callable[[FrozenDict, Dict[str, jnp.ndarray]], jnp.ndarray]
MetricFn = Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray]
DataTupleT = Tuple[Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]]
Derivative = Tuple[str, Tuple[str, str, Callable]]
ObservableFn = Callable[[FrozenDict, Dict[str, Array]], Dict[str, Array]]


def get_observable_fn(model: StackNet, observable_key: str = None) -> ObservableFn:
    """
    Get the observable function of a `model`. If no `observable_key` is specified, values for all implemented
    observables of the model are returned.

    Args:
        model (StackNet): A `StackNet` module or a module that returns a dictionary of observables.
        observable_key (str): Observable key.

    Returns: Observable function,

    """
    if observable_key is None:
        def observable_fn(p, x):
            return model.apply(p, x)
    else:
        def observable_fn(p, x):
            return {observable_key: model.apply(p, x)[observable_key]}

    return observable_fn


def grad_obs_fn(observable_fn: ObservableFn, derivative: Derivative = None) -> ObservableFn:
    """
    Get the gradient of a single observable with respect to a single input. E.g. forces as function of energy
    would be implemented as the negative gradient of the energy w.r.t. the atomic positions, which could look
    like the following: ('F', ('E', 'R', lambda y: -y.squeeze(-3)))`. If you use different names for energy and
    positions you have to adapt the keys 'E' and 'R'. As energy outputs are of shape (1), the
    gradient computation returns an Array of shape (1, n, 3), which is why we squeeze away the extra dimension.

    Args:
        observable_fn (Callable): Observable function.
        derivative (Tuple): A tuple of the form (name: str, (observable: str, input: str, op: Callable))`, which
            calculates the derivative of the form d_observable / d_input.

    Returns: An observable function which is defined as the derivative of a single observable wrt a single input.

    """
    (n, (d1, d2, scale)) = derivative

    def grad_fn(p, x):
        grads = jax.jacrev(observable_fn, argnums=1, allow_int=True)(p, x)
        return {n: scale(grads[d1][d2])}

    return grad_fn


def get_grad_observable_fn(model: StackNet, derivatives: Tuple[Derivative]) -> ObservableFn:
    """
    Get a function that gives the gradients of observables with respect to inputs. E.g. forces as function of energy
    would be implemented as the negative gradient of the energy w.r.t. the atomic positions, which would look
    like the following: ('F', ('E', 'R', lambda y: -y.squeeze(-3)))`. As energy outputs are of shape (1), the
    gradient computation returns an Array of shape (1, n, 3), which is why we squeeze away the extra dimension.

    Args:
        model (StackNet): A `StackNet` module or a module that returns a dictionary of observables.
        derivatives (Tuple[Derivative]): The derivatives that should be calculated. Each `Derivative` is tuple of
            the form (name: str, (observable: str, input: str, op: Callable))`. Argument passed should be a tuple of
            derivatives.


    Returns: Gradient function of observables w.r.t. inputs.

    """
    grad_obs_fns = []
    for d in derivatives:
        (n, (d1, d2, scale)) = d
        obs_fn = get_observable_fn(model, d1)
        grad_obs_fns += [grad_obs_fn(obs_fn, d)]

    def all_grad_obs_fn(p, x):
        results = {}
        for fn in grad_obs_fns:
            results.update(fn(p, x))
        return results

    return all_grad_obs_fn


def get_obs_and_grad_obs_fn(model: StackNet, derivatives: Tuple[Derivative] = None) -> ObservableFn:
    """
    Get function that returns observable and gradients of observables in a single call. E.g. forces as function of
    energy would be implemented as the negative gradient of the energy w.r.t. the atomic positions, which would look
    like the following: ('F', ('E', 'R', lambda y: -y.squeeze(-3)))`. As energy outputs are of shape (1), the
    gradient computation returns an Array of shape (1, n, 3), which is why we squeeze away the extra dimension.

    Args:
        model (StackNet): A `StackNet` module or a module that returns a dictionary of observables.
        derivatives (Tuple[Derivative]): The derivatives that should be calculated. Each `Derivative` is tuple of
            the form (name: str, (observable: str, input: str, op: Callable))`. Argument passed should be a tuple of
            derivatives.

    Returns: Function that maps inputs (Dict) to observables (Dict).

    """

    def merge_dicts(x, y):
        x.update(y)
        return x
    obs_fn = get_observable_fn(model)

    if derivatives is None:
        return obs_fn
    else:
        obs_grad_fn = get_grad_observable_fn(model, derivatives=derivatives)
        return lambda p, x: merge_dicts(obs_fn(p, x), obs_grad_fn(p, x))


def get_obs_and_force_fn(model: StackNet) -> ObservableFn:
    """
    Get an observable function from a `StackNet` that returns all direct observables and the force. For the force
    function to exist, the `StackNet` must take the atomic coordinates as input and return the energy as observable.
    The function uses the `get_obs_and_grad_obs_fn` function internally. Note, that the `StackNet` must have the
    correct property keys in order to "know" how to calculate the derivatives (i.e. which input is the coordinates
    and which output is the energy). If you apply your model to data that is named differently, use the
    `reset_prop_keys(new_prop_keys, sub_modules=True)` function of the `StackNet` module.

    Args:
        model (StackNet): An initialized `StackNet`.

    Returns: An observable function.

    """
    prop_keys = model.prop_keys
    E_key = prop_keys['energy']
    R_key = prop_keys['atomic_position']
    F_key = prop_keys['force']

    D_force = (F_key, (E_key, R_key, lambda y: -y.squeeze(-3)))
    obs_and_force_fn = get_obs_and_grad_obs_fn(model, derivatives=(D_force,))
    return obs_and_force_fn


def get_energy_force_stress_fn(model: StackNet):
    """
    Create an energy, force and stress function.

    Args:
        model (StackNet):

    Returns:

    """
    prop_keys = model.prop_keys
    E_key = prop_keys['energy']
    R_key = prop_keys['atomic_position']
    cell_key = prop_keys['unit_cell']
    st_key = prop_keys['stress']
    F_key = prop_keys['force']

    eps = jnp.zeros((3, 3))

    def energy_force_stress_fn(p, x):
        def u(_p, _x, _eps):
            _eps_symm = (_eps + _eps.T) / jnp.float32(2.0)
            _x = _x.copy()
            _x[R_key] = _x[R_key] + jnp.einsum('ij,nj->ni', _eps_symm, _x[R_key])
            _x[cell_key] = _x[cell_key] + jnp.einsum('ab,Ab->Aa', _eps_symm, _x[cell_key])
            return model.apply(_p, _x)[E_key].reshape()

        e, (x_D, stress) = jax.value_and_grad(u, argnums=(1, 2), allow_int=True)(p, x, eps)
        force = -x_D[R_key]

        return {E_key: e[..., None],
                F_key: force,
                st_key: stress}

    return energy_force_stress_fn
