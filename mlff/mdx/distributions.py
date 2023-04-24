import jax
import jax.numpy as jnp


def maxwell_boltzmann(masses, T0, rng_key):
    """
    Return momenta, following the Maxwell-Boltzmann distribution.

    Args:
        masses (Array): Atomic masses, shape: (n)
        T0 (float): Target temperature in eV
        rng_key (PRNGKey): JAX PRNGKey

    Returns:

    """

    xi = jax.random.normal(rng_key, (len(masses), 3))
    p = xi * jnp.sqrt(masses * T0)[:, None]
    return p
