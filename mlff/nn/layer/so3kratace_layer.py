import jax.numpy as jnp
import flax.linen as nn
import jax
import logging

from jax.ops import segment_sum
from functools import partial
from typing import (Any, Callable, Dict, Sequence)
from itertools import chain

from mlff.nn.base.sub_module import BaseSubModule
from mlff.masking.mask import safe_scale, safe_mask
from mlff.nn.mlp import MLP
from mlff.nn.activation_function.activation_function import silu
from mlff.sph_ops import make_l0_contraction_fn, SymmetricContraction


class So3krataceLayer(BaseSubModule):
    fb_rad_filter_features: Sequence[int]
    gb_rad_filter_features: Sequence[int]
    fb_sph_filter_features: Sequence[int]
    gb_sph_filter_features: Sequence[int]
    degrees: Sequence[int]
    max_body_order: int
    bo_features: int
    n_node_type: int
    fb_attention: str = 'conv_att'
    gb_attention: str = 'conv_att'
    fb_filter: str = 'radial_spherical'
    gb_filter: str = 'radial_spherical'
    n_heads: int = 4
    layer_normalization: bool = False
    sphc_normalization: bool = False
    final_layer: bool = False
    module_name: str = 'so3kratace_layer'

    @nn.compact
    def __call__(self,
                 x: jnp.ndarray,
                 chi: jnp.ndarray,
                 z_one_hot: jnp.ndarray,
                 rbf_ij: jnp.ndarray,
                 sph_ij: jnp.ndarray,
                 phi_r_cut: jnp.ndarray,
                 idx_i: jnp.ndarray,
                 idx_j: jnp.ndarray,
                 pair_mask: jnp.ndarray,
                 point_mask: jnp.ndarray,
                 *args,
                 **kwargs):
        """

        Args:
            x (Array): Atomic features, shape: (n,F)
            chi (Array): Spherical harmonic coordinates, shape: (n,m_tot)
            rbf_ij (Array): RBF expanded distances, shape: (n_pairs,K)
            sph_ij (Array): Spherical harmonics from i to j, shape: (n_pairs,m_tot)
            phi_r_cut (Array): Output of the cutoff function feature block, shape: (n_pairs)
            idx_i (Array): index centering atom, shape: (n_pairs)
            idx_j (Array): index neighboring atom, shape: (n_pairs)
            pair_mask (Array): index based mask to exclude pairs that come from index padding, shape: (n_pairs)
            point_mask (Array): index based mask to exclude nodes that come from padding, shape: (n)
            *args ():
            **kwargs ():

        Returns:

        """
        self.sow('record', 'chi_in', chi)

        chi_ij = safe_scale(jax.vmap(lambda i, j: chi[j] - chi[i])(idx_i, idx_j),
                            scale=pair_mask[:, None])  # shape: (P,m_tot)

        contraction_fn = make_l0_contraction_fn(self.degrees, dtype=chi.dtype)
        m_chi_ij = contraction_fn(chi_ij)  # shape: (P,|l|)

        # pre layer-normalization
        if self.layer_normalization:
            x_pre_1 = safe_mask(point_mask[:, None] != 0, fn=nn.LayerNorm(), operand=x)
        else:
            x_pre_1 = x

        x_local = FeatureBlock(filter=self.fb_filter,
                               rad_filter_features=self.fb_rad_filter_features,
                               sph_filter_features=self.fb_sph_filter_features,
                               attention=self.fb_attention,
                               n_heads=self.n_heads)(x=x_pre_1,
                                                     rbf_ij=rbf_ij,
                                                     d_chi_ij_l=m_chi_ij,
                                                     phi_r_cut=phi_r_cut,
                                                     idx_i=idx_i,
                                                     idx_j=idx_j,
                                                     pair_mask=pair_mask)  # shape: (n,F)

        chi_local = GeometricBlock(filter=self.gb_filter,
                                   rad_filter_features=self.gb_rad_filter_features,
                                   sph_filter_features=self.gb_sph_filter_features,
                                   attention=self.gb_attention,
                                   degrees=self.degrees)(chi=chi,
                                                         sph_ij=sph_ij,
                                                         x=x_pre_1,
                                                         rbf_ij=rbf_ij,
                                                         d_chi_ij_l=m_chi_ij,
                                                         phi_r_cut=phi_r_cut,
                                                         phi_chi_cut=jnp.zeros_like(phi_r_cut, dtype=phi_r_cut.dtype),
                                                         idx_i=idx_i,
                                                         idx_j=idx_j,
                                                         pair_mask=pair_mask)  # shape: (n,m_tot)

        x_bo, chi_bo = BodyOrderExpansionBlock(self.bo_features,
                                               self.degrees,
                                               self.max_body_order,
                                               self.n_node_type)(x_local, chi_local, z_one_hot)

        # add features
        x_skip_1 = x + x_local + x_bo
        chi_skip_1 = chi + chi_local + chi_bo

        # second pre layer-normalization
        if self.layer_normalization:
            x_pre_2 = safe_mask(point_mask[:, None] != 0, fn=nn.LayerNorm(), operand=x_skip_1)
        else:
            x_pre_2 = x_skip_1

        # feature <-> sphc interaction layer
        delta_x, delta_chi = InteractionBlock(self.degrees,
                                              parity=True)(x_pre_2, chi_skip_1, point_mask)

        # second skip connection
        x_skip_2 = (x_skip_1 + delta_x)
        chi_skip_2 = (chi_skip_1 + delta_chi)

        # in the final layer apply post layer-normalization
        if self.final_layer:
            if self.layer_normalization:
                x_skip_2 = safe_mask(point_mask[:, None] != 0, fn=nn.LayerNorm(), operand=x_skip_2)
            else:
                x_skip_2 = x_skip_2

        self.sow('record', 'chi_out', chi_skip_2)

        return {'x': x_skip_2, 'chi': chi_skip_2}

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {'fb_filter': self.fb_filter,
                                   'fb_rad_filter_features': self.fb_rad_filter_features,
                                   'fb_sph_filter_features': self.fb_sph_filter_features,
                                   'fb_attention': self.fb_attention,
                                   'gb_filter': self.gb_filter,
                                   'gb_rad_filter_features': self.gb_rad_filter_features,
                                   'gb_sph_filter_features': self.gb_sph_filter_features,
                                   'gb_attention': self.gb_attention,
                                   'n_heads': self.n_heads,
                                   'degrees': self.degrees,
                                   'max_body_order': self.max_body_order,
                                   'bo_features': self.bo_features,
                                   'n_node_type': self.n_node_type,
                                   'layer_normalization': self.layer_normalization,
                                   'sphc_normalization': self.sphc_normalization,
                                   'final_layer': self.final_layer
                                   }
                }


class FeatureBlock(nn.Module):
    filter: str
    rad_filter_features: Sequence[int]
    sph_filter_features: Sequence[int]
    attention: str  # TODO: deprecated
    n_heads: int

    def setup(self):
        if self.filter == 'radial':
            self.filter_fn = InvariantFilter(n_heads=1,
                                             features=self.rad_filter_features,
                                             activation_fn=silu)
        elif self.filter == 'radial_spherical':
            self.filter_fn = RadialSphericalFilter(rad_n_heads=1,
                                                   rad_features=self.rad_filter_features,
                                                   sph_n_heads=1,
                                                   sph_features=self.sph_filter_features,
                                                   activation_fn=silu)
        else:
            msg = "Filter argument `{}` is not a valid value.".format(self.filter)
            raise ValueError(msg)

        self.attention_fn = ConvAttention(n_heads=self.n_heads)

    @nn.compact
    def __call__(self,
                 x: jnp.ndarray,
                 rbf_ij: jnp.ndarray,
                 d_chi_ij_l: jnp.ndarray,
                 phi_r_cut: jnp.ndarray,
                 idx_i: jnp.ndarray,
                 idx_j: jnp.ndarray,
                 pair_mask: jnp.ndarray,
                 *args,
                 **kwargs):
        """

        Args:
            x (Array): Atomic features, shape: (n,F)
            rbf_ij (Array): RBF expanded distances, shape: (n_pairs,K)
            d_chi_ij_l (Array): Per degree distances of SPHCs, shape: (n_all_pairs,|L|)
            phi_r_cut (Array): Output of the cutoff function, shape: (n_pairs)
            idx_i (Array): index centering atom, shape: (n_pairs)
            idx_j (Array): index neighboring atom, shape: (n_pairs)
            pair_mask (Array): index based mask to exclude pairs that come from index padding, shape: (n_pairs)
            *args ():
            **kwargs ():

        Returns:

        """
        w_ij = self.filter_fn(rbf=rbf_ij, d_gamma=d_chi_ij_l)  # shape: (n_pairs,F)
        x_ = self.attention_fn(x=x,
                               w_ij=w_ij,
                               phi_r_cut=phi_r_cut,
                               idx_i=idx_i,
                               idx_j=idx_j,
                               pair_mask=pair_mask)  # shape: (n,F)
        return x_


class GeometricBlock(nn.Module):
    degrees: Sequence[int]
    filter: str
    rad_filter_features: Sequence[int]
    sph_filter_features: Sequence[int]
    attention: str  # TODO: deprecated

    def setup(self):
        if self.filter == 'radial':
            self.filter_fn = InvariantFilter(n_heads=1,
                                             features=self.rad_filter_features,
                                             activation_fn=silu)
        elif self.filter == 'radial_spherical':
            self.filter_fn = RadialSphericalFilter(rad_n_heads=1,
                                                   rad_features=self.rad_filter_features,
                                                   sph_n_heads=1,
                                                   sph_features=self.sph_filter_features,
                                                   activation_fn=silu)
        else:
            msg = "Filter argument `{}` is not a valid value.".format(self.filter)
            raise ValueError(msg)

        self.attention_fn = SphConvAttention(n_heads=len(self.degrees), harmonic_orders=self.degrees)

    @nn.compact
    def __call__(self,
                 chi: jnp.ndarray,
                 sph_ij: jnp.ndarray,
                 x: jnp.ndarray,
                 rbf_ij: jnp.ndarray,
                 d_chi_ij_l: jnp.ndarray,
                 phi_r_cut: jnp.ndarray,
                 phi_chi_cut: jnp.ndarray,
                 idx_i: jnp.ndarray,
                 idx_j: jnp.ndarray,
                 pair_mask: jnp.ndarray,
                 *args,
                 **kwargs):
        """

        Args:
            chi (array): spherical coordinates for all orders l, shape: (n,m_tot)
            sph_ij (array): spherical harmonics for all orders l, shape: (n_all_pairs,n,m_tot)
            x (array): atomic embeddings, shape: (n,F)
            rbf_ij (array): radial basis expansion of distances, shape: (n_pairs,K)
            d_chi_ij_l (array): pairwise distance between spherical coordinates, shape: (n_all_pairs,|L|)
            phi_r_cut (array): filter cutoff, shape: (n_pairs,L)
            phi_chi_cut (array): cutoff that scales filter values based on distance in Spherical space,
                shape: (n_all_pairs,|L|)
            idx_i (Array): index centering atom, shape: (n_pairs)
            idx_j (Array): index neighboring atom, shape: (n_pairs)
            pair_mask (Array): index based mask to exclude pairs that come from index padding, shape: (n_pairs)
            *args ():
            **kwargs ():

        Returns:

        """
        w_ij = safe_scale(self.filter_fn(rbf=rbf_ij, d_gamma=d_chi_ij_l),
                          scale=pair_mask[:, None])  # shape: (P,F)
        chi_ = self.attention_fn(chi=chi,
                                 sph_ij=sph_ij,
                                 x=x,
                                 w_ij=w_ij,
                                 phi_r_cut=phi_r_cut,
                                 phi_chi_cut=phi_chi_cut,
                                 idx_i=idx_i,
                                 idx_j=idx_j,
                                 pair_mask=pair_mask)  # shape: (n,m_tot)
        return chi_  # shape: (n,m_tot)


class InteractionBlock(nn.Module):
    degrees: Sequence[int]
    parity: bool

    def setup(self):
        segment_ids = jnp.array(
            [y for y in chain(*[[n] * (2 * self.degrees[n] + 1) for n in range(len(self.degrees))])])
        num_segments = len(self.degrees)
        self.v_segment_sum = jax.vmap(partial(segment_sum, segment_ids=segment_ids, num_segments=num_segments))

        _repeats = [2 * y + 1 for y in self.degrees]
        self.repeat_fn = partial(jnp.repeat, repeats=jnp.array(_repeats), axis=-1, total_repeat_length=sum(_repeats))

        self.contraction_fn = make_l0_contraction_fn(degrees=self.degrees)

    @nn.compact
    def __call__(self, x, chi, point_mask, *args, **kwargs):
        """

        Args:
            x (Array): shape: (n,F)
            chi (Array): shape: (n,m_tot)
            point_mask (Array) shape: (n)
            *args ():
            **kwargs ():

        Returns:

        """
        F = x.shape[-1]
        nl = len(self.degrees)

        d_chi = self.contraction_fn(chi)  # shape: (n,|l|)

        y = jnp.concatenate([x, d_chi], axis=-1)  # shape: (n,F+|l|)
        a1, b1 = jnp.split(MLP(features=[int(F + nl)],
                               activation_fn=silu)(y),
                           indices_or_sections=[F], axis=-1)
        # shape: (n,F) / shape: (n,n_l) / shape: (n,n_l)
        return a1, self.repeat_fn(b1) * chi


class BodyOrderExpansionBlock(nn.Module):
    features: int
    degrees: Sequence[int]
    max_body_order: int
    n_node_type: int

    def setup(self):
        contraction_fn = make_l0_contraction_fn(self.degrees)
        self.contraction_fn = jax.vmap(contraction_fn, in_axes=2, out_axes=2)

    @nn.compact
    def __call__(self, x, chi, z_one_hot, *args, **kwargs):
        """

        Args:
            x (Array): node features, shape: (n,F)
            chi (Array): node spherical harmonic coordinates, (n,m_tot)
            *args ():
            **kwargs ():

        Returns:

        """
        x_chi = nn.Dense(self.features)(x)[:, None, :] * chi[:, :, None]  # shape: (n,m_tot,F_proj)
        # TODO: replace shared dense by one dense per degree
        x_chi = nn.Dense(self.features, use_bias=False)(x_chi)  # shape: (n,m_tot,F_proj)
        if self.max_body_order > 2:
            x_chi_bo = SymmetricContraction(degrees_out=self.degrees,
                                            degrees_in=self.degrees,
                                            n_feature=self.features,
                                            max_body_order=self.max_body_order,
                                            n_node_type=self.n_node_type)(jnp.swapaxes(x_chi, -1, -2), z_one_hot)  # shape: (n,F_proj,m_tot)
            x_chi_bo = jnp.swapaxes(x_chi_bo, -1, -2)  # shape: (n,F_proj)
        else:
            x_chi_bo = x_chi

        chi_bo = nn.Dense(1, use_bias=False)(x_chi_bo).squeeze(-1)  # shape: (n,m_tot)
        x_bo = nn.silu(nn.Dense(x.shape[-1])(self.contraction_fn(x_chi_bo).reshape(x.shape[0], -1)))  # shape: (n,F)

        return x_bo, chi_bo


class InvariantFilter(nn.Module):
    n_heads: int
    features: Sequence[int]
    activation_fn: Callable = silu

    def setup(self):
        assert self.features[-1] % self.n_heads == 0

        f_out = int(self.features[-1] / self.n_heads)
        self._features = [*self.features[:-1], f_out]
        self.filter_fn = nn.vmap(MLP,
                                 in_axes=None, out_axes=-2,
                                 axis_size=self.n_heads,
                                 variable_axes={'params': 0},
                                 split_rngs={'params': True}
                                 )

    @nn.compact
    def __call__(self, rbf, *args, **kwargs):
        """
        Filter build from invariant geometric features.

        Args:
            rbf (Array): pairwise geometric features, shape: (...,K)
            *args ():
            **kwargs ():

        Returns: filter values, shape: (...,F)

        """
        w = self.filter_fn(self._features, self.activation_fn)(rbf)  # shape: (...,n_heads,F_head)
        w = w.reshape(*rbf.shape[:-1], -1)  # shape: (...,n,F)
        return w


class RadialSphericalFilter(nn.Module):
    rad_n_heads: int
    rad_features: Sequence[int]
    sph_n_heads: int
    sph_features: Sequence[int]
    activation_fn: Callable = silu

    def setup(self):
        assert self.rad_features[-1] % self.rad_n_heads == 0
        assert self.sph_features[-1] % self.sph_n_heads == 0

        f_out_rad = int(self.rad_features[-1] / self.rad_n_heads)
        f_out_sph = int(self.sph_features[-1] / self.sph_n_heads)

        self._rad_features = [*self.rad_features[:-1], f_out_rad]
        self._sph_features = [*self.sph_features[:-1], f_out_sph]

        self.rad_filter_fn = nn.vmap(MLP,
                                     in_axes=None, out_axes=-2,
                                     axis_size=self.rad_n_heads,
                                     variable_axes={'params': 0},
                                     split_rngs={'params': True}
                                     )

        self.sph_filter_fn = nn.vmap(MLP,
                                     in_axes=None, out_axes=-2,
                                     axis_size=self.sph_n_heads,
                                     variable_axes={'params': 0},
                                     split_rngs={'params': True}
                                     )

    @nn.compact
    def __call__(self, rbf, d_gamma, *args, **kwargs):
        """
        Filter build from invariant geometric features.

        Args:
            rbf (Array): pairwise, radial basis expansion, shape: (...,K)
            d_gamma (Array): pairwise distance of spherical coordinates, shape: (...,n_l)
            *args ():
            **kwargs ():

        Returns: filter values, shape: (...,F)

        """
        w = self.rad_filter_fn(self._rad_features, self.activation_fn)(rbf)  # shape: (...,n_heads,F_head)
        w += self.sph_filter_fn(self._sph_features, self.activation_fn)(d_gamma)  # shape: (...,n_heads,F_head)
        w = w.reshape(*rbf.shape[:-1], -1)  # shape: (...,n,n,F)
        return w


class ConvAttention(nn.Module):
    n_heads: int

    def setup(self):
        self.coeff_fn = nn.vmap(ConvAttentionCoefficients,
                                in_axes=(-2, -2, None, None), out_axes=-1,
                                axis_size=self.n_heads,
                                variable_axes={'params': 0},
                                split_rngs={'params': True}
                                )

        self.aggregate_fn = nn.vmap(AttentionAggregation,
                                    in_axes=(-2, -1, None, None), out_axes=-2,
                                    axis_size=self.n_heads,
                                    variable_axes={'params': 0},
                                    split_rngs={'params': True}
                                    )

    @nn.compact
    def __call__(self, x, w_ij, phi_r_cut, idx_i, idx_j, pair_mask, *args, **kwargs):
        """

        Args:
            x (Array): atomic embeddings, shape: (n,F)
            w_ij (Array): filter, shape: (n_pairs,F)
            phi_r_cut (Array): cutoff that scales attention coefficients, shape: (n_pairs)

        Returns:

        """
        inv_x_head_split, x_heads = equal_head_split(x, n_heads=self.n_heads)  # shape: (n,n_heads,F_head)
        _, w_heads = equal_head_split(w_ij, n_heads=self.n_heads)  # shape: (n_pairs,n_heads,F_head)
        alpha = self.coeff_fn()(x_heads, w_heads, idx_i, idx_j)  # shape: (n_pairs,n_heads)
        alpha = safe_scale(alpha, scale=pair_mask[:, None] * phi_r_cut[:, None])  # shape: (n_pairs,n_heads)

        # save attention values for later analysis
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        self.sow('record', 'alpha', alpha)
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        x_ = inv_x_head_split(self.aggregate_fn()(x_heads, alpha, idx_i, idx_j))  # shape: (n,F)
        return x_


class SphConvAttention(nn.Module):
    n_heads: int
    harmonic_orders: Sequence[int]

    def setup(self):
        _repeats = [2 * y + 1 for y in self.harmonic_orders]
        self.repeat_fn = partial(jnp.repeat, repeats=jnp.array(_repeats), axis=-1, total_repeat_length=sum(_repeats))
        self.coeff_fn = nn.vmap(ConvAttentionCoefficients,
                                in_axes=(-2, -2, None, None), out_axes=-1,
                                axis_size=self.n_heads,
                                variable_axes={'params': 0},
                                split_rngs={'params': True}
                                )

    @nn.compact
    def __call__(self, chi, sph_ij, x, w_ij, phi_r_cut, phi_chi_cut, idx_i, idx_j, pair_mask, *args, **kwargs):
        """

        Args:
            chi (Array): spherical coordinates for all degrees l, shape: (n,m_tot)
            sph_ij (Array): spherical harmonics for all degrees l, shape: (n_pairs,m_tot)
            x (Array): atomic embeddings, shape: (n,F)
            w_ij (Array): filter, shape: (n_pairs,F)
            phi_r_cut (Array): cutoff that scales attention coefficients, shape: (n_pairs)
            phi_chi_cut (Array): cutoff that scales filter values based on distance in spherical space,
                shape: (n_pairs,n_l)
            idx_i (Array): index centering atom, shape: (n_pairs)
            idx_j (Array): index neighboring atom, shape: (n_pairs)
            pair_mask (Array): index based mask to exclude pairs that come from index padding, shape: (n_pairs)
            args:
            kwargs:

        Returns:

        """

        # number of heads equals number of harmonics, i.e. n_heads = n_l
        inv_x_head_split, x_heads = equal_head_split(x, n_heads=self.n_heads)  # shape: (n,n_heads,F_head)
        _, w_ij_heads = equal_head_split(w_ij, n_heads=self.n_heads)  # shape: (n_pairs,n_heads,F_head)
        alpha_ij = self.coeff_fn()(x_heads, w_ij_heads, idx_i, idx_j)  # shape: (n_pairs,n_heads)
        alpha_r_ij = safe_scale(alpha_ij, scale=pair_mask[:, None] * phi_r_cut[:, None])  # shape: (n_pairs,n_heads)
        alpha_s_ij = safe_scale(alpha_ij, scale=pair_mask[:, None] * phi_chi_cut[:, None])  # shape: (n_pairs,n_heads)
        alpha_ij = alpha_r_ij + alpha_s_ij  # shape: (n_pairs,n_heads)

        # save attention values for later analysis
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        self.sow('record', 'alpha_r', alpha_r_ij)
        self.sow('record', 'alpha_s', alpha_s_ij)
        self.sow('record', 'alpha', alpha_ij)
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        alpha_ij = self.repeat_fn(alpha_ij)  # shape: (n_pairs,m_tot)
        chi_ = segment_sum(alpha_ij * sph_ij, segment_ids=idx_i, num_segments=x.shape[0])  # shape: (n,m_tot)
        return chi_


class ConvAttentionCoefficients(nn.Module):
    @nn.compact
    def __call__(self, x, w_ij, idx_i, idx_j):
        """

        Args:
            x (Array): atomic embeddings, shape: (n,F)
            w_ij (Array): filter, shape: (n_pairs,F)
            idx_i (Array): index centering atom, shape: (n_pairs)
            idx_j (Array): index neighboring atom, shape: (n_pairs)

        Returns: Geometric attention coefficients, shape: (n_pairs)

        """

        q_i = nn.Dense(x.shape[-1], use_bias=False)(x)[idx_i]  # shape: (n_pairs,F)
        k_j = nn.Dense(x.shape[-1], use_bias=False)(x)[idx_j]  # shape: (n_pairs,F)

        return (q_i * w_ij * k_j).sum(axis=-1) / jnp.sqrt(x.shape[-1])


class AttentionAggregation(nn.Module):
    @nn.compact
    def __call__(self, x: jnp.ndarray,
                 alpha_ij: jnp.ndarray,
                 idx_i: jnp.ndarray,
                 idx_j: jnp.ndarray) -> jnp.ndarray:
        """

        Args:
            x (Array): atomic embeddings, shape: (n,F)
            alpha_ij (Array): attention coefficients, shape: (n_pairs)
            idx_i (Array): index centering atom, shape: (n_pairs)
            idx_j (Array): index neighboring atom, shape: (n_pairs)

        Returns:

        """

        v_j = nn.Dense(x.shape[-1], use_bias=False)(x)[idx_j]  # shape: (n_pairs,F)
        return segment_sum(alpha_ij[:, None] * v_j, segment_ids=idx_i, num_segments=x.shape[0])  # shape: (n,F)


def equal_head_split(x: jnp.ndarray, n_heads: int) -> (Callable, jnp.ndarray):
    def inv_split(inputs):
        return inputs.reshape(*x.shape[:-1], -1)

    return inv_split, x.reshape(*x.shape[:-1], n_heads, -1)
