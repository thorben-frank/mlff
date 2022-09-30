import jax
import jax.numpy as jnp

from mlff.src.masking.mask import safe_mask
from mlff.src.cutoff_function.pbc import pbc_diff


@jax.jit
def inner_product_matrix(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate the scalar products between two sequences of length n of vectors of dimension d.

    Args:
        x (array_like): first sequence of vectors, shape: (...,n,d)
        y (array_like): second sequence of vectors, shape: (...,n,d)

    Returns: scalar product matrix between the vectors in the sequences, shape: (...,n,n)

    """
    n = x.shape[-2]
    return ((jnp.repeat(x, repeats=n, axis=-2) * jnp.tile(y, reps=(n, 1))).reshape((y.shape[:-2] + (n, n, -1)))).sum(-1)


@jax.jit
def coordinates_to_distance_vectors(x: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate the pairwise distance vector matrix from the coordinates in d dimensions for n points.

    Args:
        x (array_like): coordinates, shape: (...,n,d)

    Returns: distance vector matrix between coordinates, shape:  (...,n,n,d)

    """
    n = x.shape[-2]
    return (jnp.repeat(x, repeats=n, axis=-2) - jnp.tile(x, reps=(n, 1))).reshape((x.shape[:-2] + (n, n, -1)))


@jax.jit
def coordinates_to_distance_vectors_pbc(x: jnp.ndarray, lat: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate the pairwise distance vector matrix from the coordinates in d dimensions for n points.

    Args:
        x (array_like): coordinates, shape: (...,n,d)
        lat (Array): Matrix containing lattice vectors as columns, shape: (3,3)

    Returns: distance vector matrix between coordinates, shape:  (...,n,n,d)

    """
    return pbc_diff(coordinates_to_distance_vectors(x), lat=lat)


@jax.jit
def coordinates_to_distance_vectors_normalized(x):
    """
    Calculate the normalized pairwise distance vectors from the coordinates in d dimensions for n points.

    Args:
        x (Array): coordinates, shape: (...,n,d)

    Returns: normalized distance vector matrix between coordinates, shape:  (...,n,n,d)

    """

    ds = coordinates_to_distance_vectors(x)  # shape: (...,n,n,d)
    d = jnp.sum(ds**2, axis=-1, keepdims=True)
    d = safe_mask(d > 0., jnp.sqrt, d)  # shape: (...,n,n,1)

    return safe_mask(d != 0, lambda y: y / d, ds, 0)


@jax.jit
def coordinates_to_distance_matrix(x: jnp.ndarray) -> jnp.ndarray:
    """
    Calculate the pairwise distance matrix from the coordinates in d dimensions for n points.

    Args:
        x (array_like): coordinates, shape: (...,n,d)

    Returns: distance vector matrix between coordinates, shape:  (...,n,n,1)

    """

    D = jnp.sum(coordinates_to_distance_vectors(x)**2, axis=-1, keepdims=True)
    return safe_mask(D > 0., jnp.sqrt, D)
