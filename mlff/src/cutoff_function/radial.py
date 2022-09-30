import jax.numpy as jnp
import logging
from typing import Callable

from mlff.src.masking.mask import safe_mask


def get_cutoff_fn(key: str) -> Callable:
    if key == "cosine_cutoff_fn":
        return cosine_cutoff_fn
    elif key == "phys_cutoff_fn":
        return phys_cutoff_fn
    elif key == "dimenet_cutoff_fn":
        return dime_net_cutoff_fn
    elif key == "polynomial_cutoff_fn":
        return polynomial_cutoff_fn
    elif key == "exponential_cutoff_fn":
        return exponential_cutoff_fn
    else:
        logging.warning('No cutoff function set.')
        return lambda _, r_cut: jnp.ones(1)  # indeed a bit hacky think of replacement


def phys_cutoff_fn(r: jnp.ndarray, r_cut: float) -> jnp.ndarray:
    """
    Cutoff function used in PhysNet.

    Args:
        r (array): Distances, shape: (...)
        r_cut (float): Cutoff distance

    Returns: Cutoff fn output with value=0 for r > r_cut, shape (...)

    """
    cut_fn = lambda x: jnp.ones_like(x) - 6 * (x / r_cut) ** 5 + 15 * (x / r_cut) ** 4 - 10 * (x / r_cut) ** 3
    return safe_mask(r < r_cut, cut_fn, r, 0)


def cosine_cutoff_fn(r: jnp.ndarray, r_cut: float):
    """
    Behler-style cosine cutoff function.

    Args:
        r (array): Distances, shape: (...)
        r_cut (float): Cutoff distance

    Returns: Cutoff fn output with value=0 for r > r_cut, shape (...)

    """
    cut_fn = lambda x: 0.5 * (jnp.cos(x * jnp.pi / r_cut) + 1.0)
    return safe_mask(r < r_cut, cut_fn, r, 0)


def dime_net_cutoff_fn(r: jnp.ndarray, r_cut: float) -> jnp.ndarray:
    """
    Cutoff function used in DimeNet.

    Args:
        r (array): Distances, shape: (...)
        r_cut (float): Cutoff distance

    Returns: Cutoff fn output with value=0 for r > r_cut, shape (...)

    """

    return polynomial_cutoff_fn(r, r_cut, p=6)


def polynomial_cutoff_fn(r: jnp.ndarray, r_cut: float, p: int) -> jnp.ndarray:
    """
    Polynomial cutoff function.

    Args:
        r (array): Distances, shape: (...)
        r_cut (float): Cutoff distance
        p (int): Maximal order of the polynomial

    Returns: Cutoff fn output with value=0 for r > r_cut, shape (...)

    """
    pol_fn = lambda x: 1 - (1/2) * (p+1)*(p+2) * (x/r_cut)**p + p*(p+2)*(x/r_cut)**(p+1)-(1/2)*p*(p+1)*(x/r_cut)**(p+2)
    return safe_mask(r < r_cut, pol_fn, r, 0)


def exponential_cutoff_fn(r: jnp.ndarray, r_cut: float) -> jnp.ndarray:
    f_cut = lambda x: jnp.exp(-x ** 2 / ((r_cut - x) * (r_cut + x)))
    return safe_mask(mask=r < r_cut, fn=f_cut, operand=r, placeholder=0)
