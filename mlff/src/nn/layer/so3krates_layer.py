import jax.numpy as jnp
import flax.linen as nn
import jax
import logging

from jax.ops import segment_sum
from functools import partial
from typing import (Any, Callable, Dict, Sequence)
from itertools import chain

from mlff.src.nn.base.sub_module import BaseSubModule
from mlff.src.masking.mask import safe_scale, safe_mask
from mlff.src.sph_ops.base import to_scalar_feature_norm, make_degree_norm_fn
from mlff.src.nn.mlp import MLP
from mlff.src.nn.layer_norm import SPHCLayerNorm
from mlff.src.nn.activation_function.activation_function import silu
from mlff.src.cutoff_function.radial import polynomial_cutoff_fn
from mlff.src.sph_ops.contract import init_clebsch_gordan_matrix, init_expansion_fn


class So3kratesLayer(BaseSubModule):
    fb_filter: str
    fb_rad_filter_features: Sequence[int]
    fb_sph_filter_features: Sequence[int]
    fb_attention: str
    gb_filter: str
    gb_rad_filter_features: Sequence[int]
    gb_sph_filter_features: Sequence[int]
    gb_attention: str
    degrees: Sequence[int]
    n_heads: int
    chi_cut: float = None
    chi_cut_dynamic: bool = False
    parity: bool = False
    cg_path: str = None
    feature_layer_norm: bool = False
    sphc_layer_norm: bool = False
    module_name: str = 'so3krates_layer'

    def setup(self):
        _degree_norm_fn = make_degree_norm_fn(degrees=self.degrees)
        self.norm_per_degree = jax.vmap(_degree_norm_fn)

        if self.chi_cut:
            self.chi_cut_fn = partial(polynomial_cutoff_fn,
                                      r_cut=self.chi_cut,
                                      p=6)
        elif self.chi_cut_dynamic:
            self.chi_cut_fn = partial(polynomial_cutoff_fn,
                                      p=6)
        else:
            self.chi_cut_fn = lambda y, *args, **kwargs: jnp.zeros(1)

        if self.chi_cut is not None or self.chi_cut_dynamic is True:
            logging.warning('Localization in SPHC is used: Make sure that your neighborhood lists are global. Future'
                            ' implementation will work with two different index lists, but for now make sure you pass '
                            'global neighborhood lists for things to work correctly.')

        if self.chi_cut is not None and self.chi_cut_dynamic is True:
            msg = "chi_cut_dynamic is set to True and a manual chi_cut is specified. Use only one of the two."
            raise ValueError(msg)

    @nn.compact
    def __call__(self,
                 x: jnp.ndarray,
                 chi: jnp.ndarray,
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

        chi_ij = safe_scale(jax.vmap(lambda i, j: chi[i] - chi[j])(idx_i, idx_j), scale=pair_mask[:, None])
        # shape: (n_pairs,m_tot)

        d_chi_ij_l = safe_scale(self.norm_per_degree(chi_ij), pair_mask[:, None])  # shape: (n_pairs,|L|)

        d_chi_ij = safe_scale(to_scalar_feature_norm(chi_ij, axis=-1), pair_mask)  # shape: (n_pairs)

        def segment_softmax(y):
            y_ = safe_scale(y - jax.ops.segment_max(y, segment_ids=idx_i, num_segments=x.shape[0])[idx_i],
                            scale=pair_mask,
                            placeholder=0)
            a = jnp.exp(y_)
            b = segment_sum(jnp.exp(y_), segment_ids=idx_i, num_segments=x.shape[0])
            return a/b[idx_i]

        if self.chi_cut_dynamic:
            n_atoms = point_mask.sum(axis=-1)
            r_cut = 1. / n_atoms
            phi_chi_cut = safe_scale(self.chi_cut_fn(segment_softmax(d_chi_ij), r_cut=r_cut),
                                     scale=pair_mask)
        else:
            phi_chi_cut = safe_scale(self.chi_cut_fn(segment_softmax(d_chi_ij)), scale=pair_mask)
            # TODO: test SPHC cut
            # shape: (n_pairs)

        x_ = FeatureBlock(filter=self.fb_filter,
                          rad_filter_features=self.fb_rad_filter_features,
                          sph_filter_features=self.fb_sph_filter_features,
                          attention=self.fb_attention,
                          n_heads=self.n_heads)(x=x,
                                                rbf_ij=rbf_ij,
                                                d_chi_ij_l=d_chi_ij_l,
                                                phi_r_cut=phi_r_cut,
                                                idx_i=idx_i,
                                                idx_j=idx_j,
                                                pair_mask=pair_mask)  # shape: (n,F)
        if self.feature_layer_norm:
            x_ = safe_mask(point_mask[:, None] != 0, fn=nn.LayerNorm(), operand=x_, placeholder=0)
            #x_ = nn.LayerNorm(use_bias=False)(x_)

        chi_ = GeometricBlock(filter=self.gb_filter,
                              rad_filter_features=self.gb_rad_filter_features,
                              sph_filter_features=self.gb_sph_filter_features,
                              attention=self.gb_attention,
                              degrees=self.degrees)(chi=chi,
                                                    sph_ij=sph_ij,
                                                    x=x,
                                                    rbf_ij=rbf_ij,
                                                    d_chi_ij_l=d_chi_ij_l,
                                                    phi_r_cut=phi_r_cut,
                                                    phi_chi_cut=phi_chi_cut,
                                                    idx_i=idx_i,
                                                    idx_j=idx_j,
                                                    pair_mask=pair_mask)  # shape: (n,m_tot)
        if self.sphc_layer_norm:
            chi_ = SPHCLayerNorm(degrees=self.degrees)(chi_, point_mask)

        x_, chi_ = InteractionBlock(self.degrees, parity=self.parity)(x_, chi_)
        if self.feature_layer_norm:
            x_ = safe_mask(point_mask[:, None] != 0, fn=nn.LayerNorm(), operand=x_, placeholder=0)
            #x_ = nn.LayerNorm(use_bias=False)(x_)
        if self.sphc_layer_norm:
            chi_ = SPHCLayerNorm(degrees=self.degrees)(chi_, point_mask)

        self.sow('record', 'chi_out', chi_)

        return {'x': x_, 'chi': chi_}

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
                                   'chi_cut': self.chi_cut,
                                   'chi_cut_dynamic': self.chi_cut_dynamic,
                                   'degrees': self.degrees,
                                   'parity': self.parity,
                                   'feature_layer_norm': self.feature_layer_norm,
                                   'sphc_layer_norm': self.sphc_layer_norm
                                   }
                }


class FeatureBlock(nn.Module):
    filter: str
    rad_filter_features: Sequence[int]
    sph_filter_features: Sequence[int]
    attention: str
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

        if self.attention == 'conv_att':
            self.attention_fn = ConvAttention(n_heads=self.n_heads)
        elif self.attention == 'self_att':
            self.attention_fn = SelfAttention(n_heads=self.n_heads)
        else:
            msg = "Attention argument `{}` is not a valid value.".format(self.filter)
            raise ValueError(msg)

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
        x_ = x + self.attention_fn(x=x,
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
    attention: str

    def setup(self):
        if self.filter == 'radial':
            self.filter_fn = InvariantFilter(n_heads=1,
                                             features=self.filter_features,
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

        if self.attention == 'conv_att':
            self.attention_fn = SphConvAttention(n_heads=len(self.degrees), harmonic_orders=self.degrees)
        elif self.attention == 'self_att':
            self.attention_fn = SphSelfAttention(n_heads=len(self.degrees), harmonic_orders=self.degrees)
        else:
            msg = "Attention argument `{}` is not a valid value.".format(self.filter)
            raise ValueError(msg)

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
        w_ij = safe_scale(self.filter_fn(rbf=rbf_ij, d_gamma=d_chi_ij_l), scale=pair_mask[:, None])  # shape: (n_pairs,F)
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
        segment_ids = jnp.array([y for y in chain(*[[n] * (2 * self.degrees[n] + 1) for n in range(len(self.degrees))])])
        num_segments = len(self.degrees)
        self.v_segment_sum = jax.vmap(partial(segment_sum, segment_ids=segment_ids, num_segments=num_segments))
        self.selfmix = SelfMixLayer(self.degrees, parity=self.parity)

        _repeats = [2 * y + 1 for y in self.degrees]
        self.repeat_fn = partial(jnp.repeat, repeats=jnp.array(_repeats), axis=-1, total_repeat_length=sum(_repeats))

    @nn.compact
    def __call__(self, x, chi, *args, **kwargs):
        """

        Args:
            x (): shape: (n,F)
            chi (Array): shape: (n,m_tot)
            *args ():
            **kwargs ():

        Returns:

        """
        F = x.shape[-1]
        nl = len(self.degrees)

        d_gamma = self.v_segment_sum(chi ** 2)  # shape: (n,n_l)
        d_gamma = safe_mask(d_gamma > 0, jnp.sqrt, d_gamma, 0)  # shape: (n,n_l)

        chi_nl = self.selfmix(chi)
        d_chi_nl = self.v_segment_sum(chi_nl ** 2)  # shape: (n,n_l)
        d_chi_nl = safe_mask(d_chi_nl > 0, jnp.sqrt, d_chi_nl, 0)  # shape: (n,n_l)

        y = jnp.concatenate([x, d_gamma], axis=-1)  # shape: (n,F+n_l)
        a1, a2, b1, b2 = jnp.split(MLP(features=[int(2*F + 2*nl)])(y), indices_or_sections=[F, int(2*F), int(2*F+nl)], axis=-1)
        # shape: (n,F) / shape: (n,F) / shape: (n,n_l) / shape: (n,n_l)
        a3 = MLP(features=[F])(d_chi_nl)  # shape: (n,F)
        return 1/jnp.sqrt(3)*(x + a1 + a2 * a3), chi + self.repeat_fn(b1) * chi + self.repeat_fn(b2) * chi_nl


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


class SelfMixLayer(nn.Module):
    """
    SelfMix layer as implemented in PhisNet but only trainable scalars.
    """
    harmonic_orders: Sequence[int]
    parity: bool

    def setup(self):
        _l_out_max = max(self.harmonic_orders)
        _cg = init_clebsch_gordan_matrix(degrees=self.harmonic_orders, l_out_max=_l_out_max)
        self.expansion_fn = init_expansion_fn(degrees=self.harmonic_orders, cg=_cg)

        # _repeats = [2 * y + 1 for y in self.harmonic_orders]
        # _repeat_fn = partial(jnp.repeat, repeats=jnp.array(_repeats), axis=-1, total_repeat_length=sum(_repeats))
        # _c = jnp.array([len(self.harmonic_orders)-y if y > 0 else 1 for y in self.harmonic_orders])
        # # as we are only considering the case of l1 > l2, the zeroth orders has not n_l possibilities for contraction
        # # but none. If one were to include the case of l1=l2 which means k=0 in jnp.triu() function, then one has to
        # # remove the if statement and give l3=0 the normalization factor of 1/4.
        # self.c = _repeat_fn(1./_c)  # shape: (m_tot)

        _nl = len(self.harmonic_orders)
        _repeats = [2 * y + 1 for y in self.harmonic_orders]
        self.repeat_fn = partial(jnp.repeat, repeats=jnp.array(_repeats), axis=-3, total_repeat_length=sum(_repeats))
        self.coefficients = self.param('params', nn.initializers.normal(stddev=.1), (_nl, _nl, _nl))
        # shape: (n_l,n_l,n_l)
        if self.parity:
            p = (-1) ** jnp.arange(min(self.harmonic_orders), max(self.harmonic_orders)+1)
            pxp = jnp.einsum('i,j->ij', p, p)[None, ...]  # shape: (1, n_l, n_l)
            lxl_parity_filter = (pxp == p[..., None, None])  # shape: (n_l, n_l, n_l)
            self.parity_filter = jnp.repeat(lxl_parity_filter,
                                            repeats=jnp.array(_repeats),
                                            axis=0,
                                            total_repeat_length=sum(_repeats)).astype(int)
            # shape: (m_tot, n_l, n_l)
        else:
            self.parity_filter = jnp.ones((sum(_repeats), _nl, _nl))

    def __call__(self, gamma, *args, **kwargs):
        """
        Spherical non-linear layer which mixes all valid combinations of the harmonic orders.

        Args:
            gamma (Array): spherical coordinates, shape: (n,m_tot)
            *args ():
            **kwargs ():

        Returns:

        """
        gxg = jnp.einsum('...nm, ...nl -> ...nml', gamma, gamma)[:, None, :, :]  # shape: (n, 1, m_tot, m_tot)
        jm = self.expansion_fn(gxg)*self.parity_filter  # shape: (n,m_tot,n_l,n_l)
        jm = jm * self.repeat_fn(self.coefficients)  # shape: (n,m_tot,n_l,n_l)
        gamma_ = jnp.triu(jm, k=1).sum(axis=(-2, -1))
        return gamma_


def equal_head_split(x: jnp.ndarray, n_heads: int) -> (Callable, jnp.ndarray):
    def inv_split(inputs):
        return inputs.reshape(*x.shape[:-1], -1)
    return inv_split, x.reshape(*x.shape[:-1], n_heads, -1)


class SelfAttention(nn.Module):
    n_heads: int

    def setup(self):
        self.coeff_fn = nn.vmap(AttentionCoefficients,
                                in_axes=(-2, None, None), out_axes=-1,
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
    def __call__(self, x, phi_r_cut, idx_i, idx_j, pair_mask, *args, **kwargs):
        """

        Args:
            x (Array): atomic embeddings, shape: (n,F)
            phi_r_cut (Array): cutoff that scales attention coefficients, shape: (n_pairs)
            idx_i (Array): index centering atom, shape: (n_pairs)
            idx_j (Array): index neighboring atom, shape: (n_pairs)
            pair_mask (Array): index based mask to exclude pairs that come from index padding, shape: (n_pairs)
            kwargs:

        Returns:

        """
        inv_head_split, x_heads = equal_head_split(x, n_heads=self.n_heads)  # shape: (n,n_heads,F_head)
        alpha = self.coeff_fn()(x_heads, idx_j, idx_j)  # shape: (n_pairs,n_heads)
        alpha = safe_scale(alpha, scale=pair_mask[:, None]*phi_r_cut[:, None])  # shape: (n_pairs,n_heads)
        # Note: here is scaling with pair_mask not really necessary, since phi_r_cut has been already scaled by it.
        #       However, for completeness, we do it here again.

        # save attention values for later analysis
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        self.sow('record', 'alpha', alpha)
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        x_ = inv_head_split(self.aggregate_fn()(x_heads, alpha, idx_i, idx_j))  # shape: (n,F)
        return x_


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
        alpha = safe_scale(alpha, scale=pair_mask[:, None]*phi_r_cut[:, None])  # shape: (n_pairs,n_heads)

        # save attention values for later analysis
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        self.sow('record', 'alpha', alpha)
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        x_ = inv_x_head_split(self.aggregate_fn()(x_heads, alpha, idx_i, idx_j))  # shape: (n,F)
        return x_


class SphSelfAttention(nn.Module):
    n_heads: int
    harmonic_orders: Sequence[int]

    def setup(self):
        _repeats = [2 * y + 1 for y in self.harmonic_orders]
        self.repeat_fn = partial(jnp.repeat, repeats=jnp.array(_repeats), axis=-1, total_repeat_length=sum(_repeats))
        self.coeff_fn = nn.vmap(AttentionCoefficients,
                                in_axes=(-2, None, None), out_axes=-1,
                                axis_size=self.n_heads,
                                variable_axes={'params': 0},
                                split_rngs={'params': True}
                                )

    @nn.compact
    def __call__(self,
                 chi: jnp.ndarray,
                 sph_ij: jnp.ndarray,
                 x: jnp.ndarray,
                 phi_r_cut: jnp.ndarray,
                 phi_chi_cut: jnp.ndarray,
                 idx_i: jnp.ndarray,
                 idx_j: jnp.ndarray,
                 pair_mask: jnp.ndarray,
                 *args,
                 **kwargs) -> jnp.ndarray:
        """

        Args:
            chi (Array): spherical coordinates for all degrees l, shape: (n,m_tot)
            sph_ij (Array): spherical harmonics for all degrees l, shape: (n_pairs,m_tot)
            x (Array): atomic embeddings, shape: (n,F)
            phi_r_cut (Array): cutoff that scales attention coefficients, shape: (n_pairs)
            phi_chi_cut (Array): cutoff that scales filter values based on distance in Spherical space,
                shape: (n_pairs,n_l)
            idx_i (Array): index centering atom, shape: (n_pairs)
            idx_j (Array): index neighboring atom, shape: (n_pairs)
            pair_mask (Array): index based mask to exclude pairs that come from index padding, shape: (n_pairs)
            args:
            kwargs:

        Returns:

        """

        # number of heads equals number of degrees, i.e. n_heads = n_l
        inv_head_split, x_heads = equal_head_split(x, n_heads=self.n_heads)  # shape: (n,n_heads,F_head)
        alpha_ij = self.coeff_fn()(x_heads, idx_i, idx_j)  # shape: (n_pairs,n_heads)
        alpha_r_ij = safe_scale(alpha_ij, scale=pair_mask[:, None]*phi_r_cut[:, None])  # shape: (n_pairs,n_heads)
        alpha_s_ij = safe_scale(alpha_ij, scale=pair_mask[:, None]*phi_chi_cut[:, None])  # shape: (n_pairs,n_heads)
        alpha_ij = alpha_r_ij + alpha_s_ij  # shape: (n_pairs,n_heads)

        # save attention values for later analysis
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        self.sow('record', 'alpha_r', alpha_r_ij)
        self.sow('record', 'alpha_s', alpha_s_ij)
        self.sow('record', 'alpha', alpha_ij)
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        alpha_ij = self.repeat_fn(alpha_ij)  # shape: (n_pairs,m_tot)
        chi_ = chi + segment_sum(alpha_ij * sph_ij, segment_ids=idx_i, num_segments=x.shape[0])  # shape: (n,m_tot)
        return chi_


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
        alpha_r_ij = safe_scale(alpha_ij, scale=pair_mask[:, None]*phi_r_cut[:, None])  # shape: (n_pairs,n_heads)
        alpha_s_ij = safe_scale(alpha_ij, scale=pair_mask[:, None]*phi_chi_cut[:, None])  # shape: (n_pairs,n_heads)
        alpha_ij = alpha_r_ij + alpha_s_ij  # shape: (n_pairs,n_heads)

        # save attention values for later analysis
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
        self.sow('record', 'alpha_r', alpha_r_ij)
        self.sow('record', 'alpha_s', alpha_s_ij)
        self.sow('record', 'alpha', alpha_ij)
        # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

        alpha_ij = self.repeat_fn(alpha_ij)  # shape: (n_pairs,m_tot)
        chi_ = chi + segment_sum(alpha_ij * sph_ij, segment_ids=idx_i, num_segments=x.shape[0])  # shape: (n,m_tot)
        return chi_


class AttentionCoefficients(nn.Module):
    @nn.compact
    def __call__(self, x, idx_i, idx_j):
        """

        Args:
            x (Array): atomic embeddings, shape: (n,F)
            idx_i (Array): index centering atom, shape: (n_pairs)
            idx_j (Array): index neighboring atom, shape: (n_pairs)

        Returns: Geometric attention coefficients, shape: (n_pairs)

        """

        q_i = nn.Dense(x.shape[-1])(x)[idx_i]  # shape: (n_pairs,F)
        k_j = nn.Dense(x.shape[-1])(x)[idx_j]  # shape: (n_pairs,F)

        return (q_i * k_j).sum(axis=-1) / jnp.sqrt(x.shape[-1])  # shape: (n_pairs)


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

        q_i = nn.Dense(x.shape[-1])(x)[idx_i]  # shape: (n_pairs,F)
        k_j = nn.Dense(x.shape[-1])(x)[idx_j]  # shape: (n_pairs,F)

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

        v_j = nn.Dense(x.shape[-1])(x)[idx_j]  # shape: (n_pairs,F)
        return segment_sum(alpha_ij[:, None] * v_j, segment_ids=idx_i, num_segments=x.shape[0])  # shape: (n,F)
