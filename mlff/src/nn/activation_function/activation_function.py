import jax
import jax.numpy as jnp
from jax.nn import softplus, silu
from typing import Callable


def get_activation_fn(key: str) -> Callable:
    if key == "shifted_softplus":
        return shifted_softplus
    if key == "silu":
        return jax.jit(silu)


@jax.jit
def shifted_softplus(x):
    return softplus(x) - softplus(jnp.zeros(1))
