import jax.numpy as jnp
from typing import (Any, Callable, Tuple)


Array = Any


def pbc_diff(r_ij: jnp.ndarray, lat: jnp.ndarray):
    """
    Clamp differences of vectors to super cell.

    Args:
        r_ij (Array): distance vectors, shape: (...,3)
        lat (Array): matrix containing lattice vectors as columns, shape: (3,3)

    Returns: clamped distance vectors, shape: (...,3)

    """
    lat_inv = jnp.linalg.inv(lat)  # shape: (3,3)
    c = jnp.einsum('ij, ...j -> ...i', lat_inv, r_ij)  # shape: (...,3)
    delta = r_ij - jnp.einsum('ij, ...j -> ...i', lat, jnp.rint(c))  # shape: (...,3)
    return delta


# def get_pbc_fn(lat_and_inv: Tuple[Array, Array]) -> Callable[[jnp.ndarray], jnp.ndarray]:
#     """
#
#     Args:
#         lat_and_inv (Array): Tuple of 3 x 3 matrix containing lattice vectors as columns and its inverse.
#
#     Returns:
#
#     """
#     def pbc_diff_fn(r_ij: jnp.ndarray) -> jnp.ndarray:
#         """
#         Clamp differences of vectors to super cell.
#
#         Args:
#             r_ij (Array): distance vectors, shape: (...,3)
#
#         Returns: clamped distance vectors, shape: (...,3)
#
#         """
#
#         lat, lat_inv = lat_and_inv  # shape: (3,3) // shape: (3,3)
#         c = jnp.einsum('ij, ...j -> ...i', lat_inv, r_ij)  # shape: (...,3)
#         delta = r_ij - jnp.einsum('ij, ...j -> ...i', lat, jnp.rint(c))  # shape: (...,3)
#         return delta
#
#     return pbc_diff_fn
