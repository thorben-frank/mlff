from jax import numpy as jnp
import numpy as np
import flax.linen as nn
from typing import Any, Callable
import logging

from mlff.src.masking.mask import safe_mask

PlaceHolder = Any


def get_rbf_fn(key: str) -> type(nn.Module):
    if key == "rbf":
        return RBF
    if key == "phys":
        return PhysNetBasis
    if key == "bessel":
        return BesselBasis
    if key == 'bernstein':
        return BernsteinBasis


def log_factorial(n):
    """
    Logarithm of factorial for n.
    Note: Function makes use of the FORTRAN behavior of jax.numpy.sum() that gives 0 over an empty array.
    In that way, log_factorial returns 0 instead of -inf for n=0. Useful for implementing
    factorial as exp(log_factorial) as this gives automatically the correct behavior factorial(0) = 1.

    Args:
        n (int):

    Returns:

    """
    return np.sum(np.log(np.arange(1, n + 1)))


def factorial(n):
    return np.exp(log_factorial(n))


def log_binomial_coefficient(n, k):
    n_factorial = log_factorial(n)
    k_factorial = log_factorial(k)
    n_k_factorial = log_factorial(n-k)
    return n_factorial - (k_factorial+n_k_factorial)


def binomial_coefficient(n, k):
    return np.exp(log_binomial_coefficient(n, k))


class BernsteinBasis(nn.Module):
    n_rbf: int
    r_cut: float = None
    gamma_init: float = 0.9448630629184640

    def setup(self) -> None:
        b = jnp.stack(list(map(lambda x: log_binomial_coefficient(self.n_rbf - 1, x), np.arange(self.n_rbf))))
        # shape: (K)
        k = jnp.arange(self.n_rbf)  # shape: (K)
        k_rev = jnp.arange(self.n_rbf)[::-1]  # shape: (K)

        def log_bernstein_polynomial(x):
            """

            Args:
                x (Array): Array, shape: (...,1)

            Returns:

            """
            k_x = jnp.where(k != 0, k * jnp.log(x), 0)
            kk_x = jnp.where(k_rev != 0, k_rev * jnp.log(1 - x), 0)

            return b + k_x + kk_x

        self.log_bernstein_polynomial = log_bernstein_polynomial

    @nn.compact
    def __call__(self, r_ij) -> jnp.ndarray:
        """

        Args:
            r_ij (Array): shape: (...,1)

        Returns:

        """
        gamma = self.param('gamma', nn.initializers.constant(self.gamma_init, jnp.float_), ())
        exp_r = jnp.exp(-gamma*r_ij)  # shape: (...,1)

        # the cases exp_r=0 and exp_r=1 are the boundaries which give inf for the log_polynomial. They correspond
        # to the extreme cases of r_ij = 0 and r_ij = +/- inf, which are only possible in theory but might appear
        # e.g. during initialization or due to padding.
        return safe_mask(mask=((exp_r != 0) & (exp_r != 1)),
                         fn=lambda y: jnp.exp(self.log_bernstein_polynomial(y)),
                         operand=exp_r,
                         placeholder=0)


class RBF(nn.Module):
    n_rbf: int
    """
    Number of basis functions.
    """
    r_cut: float
    """
    Cutoff radius.
    """
    r_0: float = 0.

    def setup(self):
        self.offsets = jnp.linspace(self.r_0, self.r_cut, self.n_rbf)  # shape: (n_rbf)
        self.widths = jnp.abs(self.offsets[1] - self.offsets[0])  # shape: (1)
        self.coefficient = 10  # shape: (1)

    def __call__(self, r: jnp.ndarray) -> jnp.ndarray:
        """
        Expand distances in the RBF basis, used in SchNet
        (see https://proceedings.neurips.cc/paper/2017/file/303ed4c69846ab36c2904d3ba8573050-Paper.pdf)

        Args:
            r (Array): Distances, shape: (...,1)

        Returns: The expanded distances, shape: (...,n,n,L)
        """
        return jnp.exp(-self.coefficient * (r - self.offsets) ** 2)


class PhysNetBasis(nn.Module):
    n_rbf: int
    """
    Number of basis functions.
    """
    r_cut: float
    """
    Cutoff radius.
    """
    r_0: float = 0.

    def setup(self):
        self.offsets = jnp.linspace(jnp.exp(-self.r_cut), jnp.exp(-self.r_0), self.n_rbf)  # shape: (n_rbf)
        self.coefficient = ((2 / self.n_rbf) * (1 - jnp.exp(-self.r_cut))) ** (-2)  # shape: (1)

    def __call__(self, r: jnp.ndarray) -> jnp.ndarray:
        """
        Expand distances in the basis used in PhysNet (see https://arxiv.org/abs/1902.08408)

        Args:
            r (Array): Distances, shape: (...,1)

        Returns: The expanded distances, shape: (...,n,n,L)
        """
        return jnp.exp(-abs(self.coefficient) * (jnp.exp(-r) - self.offsets) ** 2)


class BesselBasis(nn.Module):
    n_rbf: int
    """
    Number of basis functions.
    """
    r_cut: float
    """
    Cutoff radius. Bessel functions are normalized within the interval [0, r_c] as well as bessel(r_c) = 0, as boundary 
    condition when solving the Helmholtz equation.
    """
    r_0: float = 0.

    def setup(self):
        self.offsets = jnp.arange(0, self.n_rbf, 1)  # shape: (n_rbf)

    def __call__(self, r: jnp.ndarray) -> jnp.ndarray:
        """
        Expand distances in the Bessel basis (see https://arxiv.org/pdf/2003.03123.pdf)

        Args:
                r (Array): Distances, shape: (...,1)

        Returns: The expanded distances, shape: (...,n,n,L)
        """

        f = lambda x: jnp.sin(jnp.pi / self.r_cut * self.offsets * x) / x
        return safe_mask(r != 0, f, r, 0)


class FourierBasis(nn.Module):
    n_rbf: int
    """
    Number of basis functions.
    """
    r_cut: float
    """
    Cutoff radius. Bessel functions are normalized within the interval [0, r_c] as well as bessel(r_c) = 0, as boundary 
    condition when solving the Helmholtz equation.
    """
    r_0: float = 0.

    def setup(self):
        self.offsets = jnp.arange(0, self.n_rbf, 1)  # shape: (n_rbf)

    def __call__(self, r: jnp.ndarray) -> jnp.ndarray:
        """
        Expand distances in the Bessel basis (see https://arxiv.org/pdf/2003.03123.pdf)

        Args:
                r (Array): Distances, shape: (...,1)

        Returns: The expanded distances, shape: (...,n,n,L)
        """

        f = lambda x: jnp.sin(jnp.pi / self.r_cut * self.offsets * x) / x
        return safe_mask(r != 0, f, r, 0)
