import jax
import jax.numpy as jnp
import flax.linen as nn
from jax.ops import segment_sum
from typing import Any, Callable, Dict
from mlff.nn.base.sub_module import BaseSubModule
from jax.scipy.special import erfc
from typing import Optional
from mlff.cutoff_function import add_cell_offsets_sparse

import jax.numpy as jnp
from jax import lax
from jax import jit
from jax import grad
from jax import vmap
from jax import random

"""
computes electrostatic energy, switches between a constant value
and the true Coulomb law between cuton and cutoff
"""


class ElectrostaticEnergy(BaseSubModule):
    prop_keys: Dict
    regression_dim: int = None
    activation_fn: Callable[[Any], Any] = lambda u: u
    output_is_zero_at_init: bool = True
    module_name: str = 'electrostatic_energy'
    partial_charges: Optional[Any] = None

    def __init__(
        self,
        ke: float = 14.399645351950548,
        cuton: float = 0.0,
        cutoff: float = 1.0,
        lr_cutoff: Optional[float] = None,
    ) -> None:
        self.ke = ke
        self.kehalf = ke / 2
        self.cuton = cuton
        self.cutoff = cutoff
        self.set_lr_cutoff(lr_cutoff)
        # should be turned on manually if the user knows what they are doing
        self.use_ewald_summation = False
        # set optional attributes to default value for jit compatibility
        self.alpha = 0.0
        self.alpha2 = 0.0
        self.two_pi = 2.0 * jnp.pi
        self.one_over_sqrtpi = 1 / jnp.sqrt(jnp.pi)
        self.kmul = jnp.array([], dtype=jnp.int32)

    def reset_parameters(self) -> None:
        """ For compatibility with other modules. """
        pass

    def set_lr_cutoff(self, lr_cutoff: Optional[float] = None) -> None:
        """ Change the long range cutoff. """
        self.lr_cutoff = lr_cutoff
        if self.lr_cutoff is not None:
            self.lr_cutoff2 = lr_cutoff ** 2
            self.two_div_cut = 2.0 / lr_cutoff
            self.rcutconstant = lr_cutoff / (lr_cutoff ** 2 + 1.0) ** (3.0 / 2.0)
            self.cutconstant = (2 * lr_cutoff ** 2 + 1.0) / (lr_cutoff ** 2 + 1.0) ** (
                3.0 / 2.0
            )
        else:
            self.lr_cutoff2 = None
            self.two_div_cut = None
            self.rcutconstant = None
            self.cutconstant = None

    def set_kmax(self, Nxmax: int, Nymax: int, Nzmax: int) -> None:
        """ Set integer reciprocal space cutoff for Ewald summation """
        kx = jnp.arange(0, Nxmax + 1)
        kx = jnp.concatenate([kx, -kx[1:]])
        ky = jnp.arange(0, Nymax + 1)
        ky = jnp.concatenate([ky, -ky[1:]])
        kz = jnp.arange(0, Nzmax + 1)
        kz = jnp.concatenate([kz, -kz[1:]])
        kmul = jnp.array(jnp.cartesian_product(kx, ky, kz)[1:])  # 0th entry is 0 0 0
        kmax = max(max(Nxmax, Nymax), Nzmax)
        self.kmul = kmul[jnp.sum(kmul ** 2, axis=-1) <= kmax ** 2]

    def set_alpha(self, alpha: Optional[float] = None) -> None:
        """ Set real space damping parameter for Ewald summation """
        if alpha is None:  # automatically determine alpha
            alpha = 4.0 / self.cutoff + 1e-3
        self.alpha = alpha
        self.alpha2 = alpha ** 2
        self.two_pi = 2.0 * jnp.pi
        self.one_over_sqrtpi = 1 / jnp.sqrt(jnp.pi)
        # print a warning if alpha is so small that the reciprocal space sum
        # might "leak" into the damped part of the real space coulomb interaction
        if alpha * self.cutoff < 4.0:  # erfc(4.0) ~ 1e-8
            print(
                "Warning: Damping parameter alpha is",
                alpha,
                "but probably should be at least",
                4.0 / self.cutoff,
            )

    def _real_space(
        self,
        N: int,
        q: jnp.ndarray,
        rij: jnp.ndarray,
        idx_i: jnp.ndarray,
        idx_j: jnp.ndarray,
    ) -> jnp.ndarray:
        fac = self.kehalf * q[idx_i] * q[idx_j]
        f = switch_function(rij, self.cuton, self.cutoff)
        coulomb = 1.0 / rij
        damped = 1.0 / (rij ** 2 + 1.0) ** (1.0 / 2.0)
        pairwise = fac * (f * damped + (1 - f) * coulomb) * jnp.erfc(self.alpha * rij)
        return jnp.zeros(N).at[idx_i].add(pairwise)

    def _reciprocal_space(
        self,
        q: jnp.ndarray,
        R: jnp.ndarray,
        cell: jnp.ndarray,
        num_batch: int,
        batch_seg: jnp.ndarray,
        eps: float = 1e-8,
    ) -> jnp.ndarray:
        box_length = jnp.diagonal(cell, axis1=-2, axis2=-1)
        k = self.two_pi * self.kmul / box_length[..., None]
        k2 = jnp.sum(k * k, axis=-1)
        qg = jnp.exp(-0.25 * k2 / self.alpha2) / k2
        dot = jnp.sum(k[batch_seg] * R[..., None], axis=-1)
        q_real = jnp.zeros((num_batch, dot.shape[-1])).at[batch_seg].add(q[..., None] * jnp.cos(dot))
        q_imag = jnp.zeros((num_batch, dot.shape[-1])).at[batch_seg].add(q[..., None] * jnp.sin(dot))
        qf = q_real ** 2 + q_imag ** 2
        e_reciprocal = (
            self.two_pi / jnp.prod(box_length, axis=1) * jnp.sum(qf * qg, axis=-1)
        )
        q2 = q * q
        e_self = self.alpha * self.one_over_sqrtpi * q2
        w = q2 + eps
        wnorm = jnp.zeros(num_batch).at[batch_seg].add(w)
        w = w / wnorm.at[batch_seg]
        e_reciprocal = w * e_reciprocal.at[batch_seg]
        return self.ke * (e_reciprocal - e_self)

    def _ewald(
        self,
        N: int,
        q: jnp.ndarray,
        R: jnp.ndarray,
        rij: jnp.ndarray,
        idx_i: jnp.ndarray,
        idx_j: jnp.ndarray,
        cell: jnp.ndarray,
        num_batch: int,
        batch_seg: jnp.ndarray,
    ) -> jnp.ndarray:
        e_real = self._real_space(N, q, rij, idx_i, idx_j)
        e_reciprocal = self._reciprocal_space(q, R, cell, num_batch, batch_seg)
        return e_real + e_reciprocal

    def _coulomb(
        self,
        N: int,
        q: jnp.ndarray,
        rij: jnp.ndarray,
        idx_i: jnp.ndarray,
        idx_j: jnp.ndarray,
    ) -> jnp.ndarray:
        fac = self.kehalf * q[idx_i] * q[idx_j]
        f = switch_function(rij, self.cuton, self.cutoff)
        if self.lr_cutoff is None:
            coulomb = 1.0 / rij
            damped = 1.0 / (rij ** 2 + 1.0) ** (1.0 / 2.0)
        else:
            coulomb = lax.where(
                rij < self.lr_cutoff,
                1.0 / rij + rij / self.lr_cutoff2 - self.two_div_cut,
                jnp.zeros_like(rij),
            )
            damped = lax.where(
                rij < self.lr_cutoff,
                1.0 / (rij ** 2 + 1.0) ** (1.0 / 2.0)
                + rij * self.rcutconstant
                - self.cutconstant,
                jnp.zeros_like(rij),
            )
        pairwise = fac * (f * damped + (1 - f) * coulomb)
        return jnp.zeros(N).at[idx_i].add(pairwise)

    @nn.compact
    def __call__(self, inputs: Dict, *args, **kwargs) -> jnp.ndarray:  
        # N: int,
        # q: jnp.ndarray,
        # rij: jnp.ndarray,
        # idx_i: jnp.ndarray,
        # idx_j: jnp.ndarray,
        # R: Optional[jnp.ndarray] = None,
        # cell: Optional[jnp.ndarray] = None,
        # num_batch: int = 1,
        # batch_seg: Optional[jnp.ndarray] = None,
        node_mask = inputs['node_mask']  # (num_nodes)
        num_nodes = len(node_mask)
        partial_charges = self.partial_charges(inputs)['partial_charges']
        rij = inputs['absolute_distance']  # shape: (num_graphs, 3, 3)
        idx_i = inputs['idx_i']  # shape: (num_pairs)
        idx_j = inputs['idx_j']  # shape: (num_pairs)
        cell = inputs.get('cell')  # shape: (num_graphs, 3, 3)
        num_batch: int = 1
        batch_segments = inputs['batch_segments']  # (num_nodes)
        input_convention: str = 'positions'
        # cell = inputs.get('cell')  # shape: (num_graphs, 3, 3)
        cell_offsets = inputs.get('cell_offset')  # shape: (num_pairs, 3)

        if input_convention == 'positions':
            positions = inputs['positions']  # (N, 3)

            # Calculate pairwise distance vectors
            r_ij = jax.vmap(
                lambda i, j: positions[j] - positions[i]
            )(idx_i, idx_j)  # (num_pairs, 3)

            # Apply minimal image convention if needed.
            if cell is not None:
                r_ij = add_cell_offsets_sparse(
                    r_ij=r_ij,
                    cell=cell,
                    cell_offsets=cell_offsets
                )  # shape: (num_pairs,3)

        if self.use_ewald_summation:
            assert positions is not None
            assert cell is not None
            assert batch_segments is not None
            return self._ewald(num_nodes, partial_charges, positions, rij, idx_i, idx_j, cell, num_batch, batch_segments)
        else:
            return self._coulomb(num_nodes, partial_charges, rij, idx_i, idx_j)
    # def forward(
    #     self,
    #     N: int,
    #     q: jnp.ndarray,
    #     rij: jnp.ndarray,
    #     idx_i: jnp.ndarray,
    #     idx_j: jnp.ndarray,
    #     R: Optional[jnp.ndarray] = None,
    #     cell: Optional[jnp.ndarray] = None,
    #     num_batch: int = 1,
    #     batch_seg: Optional[jnp.ndarray] = None,
    # ) -> jnp.ndarray:
    #     if self.use_ewald_summation:
    #         assert R is not None
    #         assert cell is not None
    #         assert batch_seg is not None
    #         return self._ewald(N, q, R, rij, idx_i, idx_j, cell, num_batch, batch_seg)
    #     else:
    #         return self._coulomb(N, q, rij, idx_i, idx_j)