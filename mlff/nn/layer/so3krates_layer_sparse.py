import jax.numpy as jnp
import flax.linen as nn
import jax
import numpy as np

from jax.ops import segment_sum
from functools import partial
from typing import (Any, Callable, Dict, Sequence)

from mlff.nn.base.sub_module import BaseSubModule
from mlff.sph_ops import make_l0_contraction_fn
from mlff.masking import mask


def split_in_heads(x: jnp.ndarray, num_heads: int) -> (Callable, jnp.ndarray):
    def inv_split(inputs):
        return inputs.reshape(*x.shape[:-1], -1)

    return inv_split, x.reshape(*x.shape[:-1], num_heads, -1)


def make_degree_repeat_fn(degrees: Sequence[int], axis: int = -1):
    repeats = np.array([2 * y + 1 for y in degrees])
    repeat_fn = partial(jnp.repeat, repeats=repeats, axis=axis, total_repeat_length=np.sum(repeats))
    return repeat_fn


class SO3kratesLayerSparse(BaseSubModule):
    degrees: Sequence[int]
    use_spherical_filter: bool
    num_heads: int = 4
    num_features_head: int = 32
    qk_non_linearity: Callable = jax.nn.silu
    residual_mlp_1: bool = False
    residual_mlp_2: bool = False
    layer_normalization_1: bool = False
    layer_normalization_2: bool = False
    message_normalization: str = 'sqrt_num_features'
    avg_num_neighbors: float = None
    activation_fn: Callable = jax.nn.silu
    behave_like_identity_fn_at_init: bool = False
    module_name: str = 'so3krates_layer_sparse'

    def setup(self):
        if self.behave_like_identity_fn_at_init:
            self.last_layer_kernel_init = jax.nn.initializers.zeros
        else:
            self.last_layer_kernel_init = jax.nn.initializers.lecun_normal()
        if self.message_normalization == 'avg_num_neighbors':
            if self.avg_num_neighbors is None:
                raise ValueError(
                    '`avg_num_neighbors` must be set for `SO3kratesLayerSparse` if using '
                    '`message_normalization=avg_num_neighbors`.'
                )

    @nn.compact
    def __call__(self,
                 x: jnp.ndarray,
                 ev: jnp.ndarray,
                 rbf_ij: jnp.ndarray,
                 ylm_ij: jnp.ndarray,
                 cut: jnp.ndarray,
                 idx_i: jnp.ndarray,
                 idx_j: jnp.ndarray,
                 *args,
                 **kwargs):
        """

        Args:
            x (Array): Node features, shape: (num_nodes, num_features)
            ev (Array): Euclidean variables, shape: (num_nodes, num_orders)
            rbf_ij (Array): RBF expanded distances, shape: (num_pairs, K)
            ylm_ij (Array): Spherical harmonics from i to j, shape: (num_pairs, num_orders)
            cut (Array): Output of the cutoff function feature block, shape: (num_pairs)
            idx_i (Array): index centering atom, shape: (num_pairs)
            idx_j (Array): index neighboring atom, shape: (num_pairs)
            *args ():
            **kwargs ():

        Returns:

        """
        num_features = x.shape[-1]

        x_att, ev_att = AttentionBlock(
            degrees=self.degrees,
            num_heads=self.num_heads,
            num_features_head=self.num_features_head,
            qk_non_linearity=self.qk_non_linearity,
            use_spherical_filter=self.use_spherical_filter,
            output_is_zero_at_init=self.behave_like_identity_fn_at_init,
            activation_fn=self.activation_fn,
            normalization=self.message_normalization,
            avg_num_neighbors=self.avg_num_neighbors,
            name='attention_block')(x=x,
                                    ev=ev,
                                    rbf_ij=rbf_ij,
                                    ylm_ij=ylm_ij,
                                    cut=cut,
                                    idx_i=idx_i,
                                    idx_j=idx_j,
                                    avg_num_neighbors=self.avg_num_neighbors)  # (num_nodes, num_features), (num_nodes, num_orders)

        x = x + x_att  # (num_nodes, num_features)
        ev = ev + ev_att  # (num_nodes, num_orders)

        if self.layer_normalization_1:
            x = nn.LayerNorm(name='layer_normalization_1')(x)  # (num_nodes, num_features)

        if self.residual_mlp_1:
            y = self.activation_fn(x)  # (num_nodes, num_features)
            y = nn.Dense(
                features=num_features,
                name='res_mlp_1_layer_1'
            )(y)  # (num_nodes, num_features)
            y = self.activation_fn(y)  # (num_nodes, num_features)
            y = nn.Dense(
                features=num_features,
                kernel_init=self.last_layer_kernel_init,
                name='res_mlp_1_layer_2'
            )(y)  # (num_nodes, num_features)
            x = x + y  # (num_nodes, num_features)

        # Interaction layer
        x_ex, ev_ex = ExchangeBlock(
            degrees=self.degrees,
            output_is_zero_at_init=self.behave_like_identity_fn_at_init,
            activation_fn=self.activation_fn,
            name='exchange_block')(x=x, ev=ev)  # (num_nodes, num_features), (num_nodes, num_orders)

        x = x + x_ex  # (num_nodes, num_features)
        ev = ev + ev_ex  # (num_nodes, num_orders)

        if self.residual_mlp_2:
            y = self.activation_fn(x)  # (num_nodes, num_features)
            y = nn.Dense(
                features=num_features,
                name='res_mlp_2_layer_1'
            )(y)  # (num_nodes, num_features)
            y = self.activation_fn(y)  # (num_nodes, num_features)
            y = nn.Dense(
                features=num_features,
                kernel_init=self.last_layer_kernel_init,
                name='res_mlp_2_layer_2'
            )(y)  # (num_nodes, num_features)
            x = x + y  # (num_nodes, num_features)

        if self.layer_normalization_2:
            x = nn.LayerNorm(name='layer_normalization_2')(x)  # (num_nodes, num_features)

        return {'x': x, 'ev': ev}  # (num_nodes, num_features), (num_nodes, num_orders)

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {'degrees': self.degrees,
                                   'use_spherical_filter': self.use_spherical_filter,
                                   'num_heads': self.num_heads,
                                   'num_features_head': self.num_features_head,
                                   'qk_non_linearity': self.qk_non_linearity,
                                   'residual_mlp_1': self.residual_mlp_1,
                                   'residual_mlp_2': self.residual_mlp_2,
                                   'layer_normalization_1': self.layer_normalization_1,
                                   'layer_normalization_2': self.layer_normalization_2,
                                   'activation_fn': self.activation_fn,
                                   'behave_like_identity_fn_at_init': self.behave_like_identity_fn_at_init
                                   }
                }


class AttentionBlock(nn.Module):
    degrees: Sequence[int]
    num_heads: int = 4
    num_features_head: int = 32
    normalization: str = 'sqrt_num_features'
    avg_num_neighbors: float = None
    qk_non_linearity: Callable = jax.nn.silu
    activation_fn: Callable = jax.nn.silu
    output_is_zero_at_init: bool = False
    use_spherical_filter: bool = True

    def setup(self):
        if self.output_is_zero_at_init:
            self.value_kernel_init = jax.nn.initializers.zeros
        else:
            self.value_kernel_init = jax.nn.initializers.lecun_normal(batch_axis=(0,))

    @nn.compact
    def __call__(self,
                 x: jnp.ndarray,
                 ev: jnp.ndarray,
                 rbf_ij: jnp.ndarray,
                 ylm_ij: jnp.ndarray,
                 cut: jnp.ndarray,
                 idx_i: jnp.ndarray,
                 idx_j: jnp.ndarray,
                 *args,
                 **kwargs):
        """

        Args:
            x (Array): Node features, shape: (num_nodes, num_features)
            ev (Array): Euclidean variables, shape: (num_nodes, num_orders)
            rbf_ij (Array): RBF expanded distances, shape: (num_pairs, K)
            ylm_ij (Array): Spherical harmonics from i to j, shape: (num_pairs, num_orders)
            cut (Array): Output of the cutoff function feature block, shape: (num_pairs)
            idx_i (Array): index centering atom, shape: (num_pairs)
            idx_j (Array): index neighboring atom, shape: (num_pairs)
            *args ():
            **kwargs ():

        Returns:

        """
        assert x.ndim == 2
        assert ev.ndim == 2
        assert idx_i.ndim == 1
        assert idx_j.ndim == 1
        assert cut.ndim == 1

        num_features = x.shape[-1]
        assert num_features % self.num_heads == 0

        # tot_num_heads = self.num_heads + len(self.degrees)
        assert num_features % len(self.degrees) == 0

        # tot_num_features = tot_num_heads * self.num_features_head

        contraction_fn = make_l0_contraction_fn(self.degrees, dtype=ev.dtype)
        degree_repeat_fn = make_degree_repeat_fn(self.degrees, axis=-1)

        w1_ij = nn.Dense(
            features=num_features,
            name='radial_filter1_layer_2'
        )(
            self.activation_fn(
                nn.Dense(
                    features=num_features,
                    name='radial_filter1_layer_1')(rbf_ij)
            )
        )  # (num_pairs, tot_num_features)

        w2_ij = nn.Dense(
            features=num_features,
            name='radial_filter2_layer_2'
        )(
            self.activation_fn(
                nn.Dense(
                    features=num_features,
                    name='radial_filter2_layer_1')(rbf_ij)
            )
        )  # (num_pairs, tot_num_features)

        ev_i = ev[idx_i]  # (num_pairs)
        ev_j = ev[idx_j]  # (num_pairs)

        if self.use_spherical_filter:
            w1_ij += nn.Dense(
                features=num_features,
                name='spherical_filter1_layer_2'
            )(
                self.activation_fn(
                    nn.Dense(
                        features=num_features // 4,
                        name='spherical_filter1_layer_1')(contraction_fn(ev_j - ev_i))
                )
            )  # (num_pairs, tot_num_features)

            w2_ij += nn.Dense(
                features=num_features,
                name='spherical_filter2_layer_2'
            )(
                self.activation_fn(
                    nn.Dense(
                        features=num_features // 4,
                        name='spherical_filter2_layer_1')(contraction_fn(ev_j - ev_i))
                )
            )  # (num_pairs, tot_num_features)

        _, w1_ij = split_in_heads(w1_ij, num_heads=self.num_heads)
        _, w2_ij = split_in_heads(w2_ij, num_heads=len(self.degrees))
        # _, (num_pairs, tot_num_heads, num_features_head)

        Wq1 = self.param(
            'Wq1',
            jax.nn.initializers.lecun_normal(batch_axis=(0,)),
            (self.num_heads, num_features // self.num_heads, num_features // self.num_heads)
        )  # (tot_num_heads, num_features_head, num_features // tot_num_heads)

        Wk1 = self.param(
            'Wk1',
            jax.nn.initializers.lecun_normal(batch_axis=(0,)),
            (self.num_heads, num_features // self.num_heads, num_features // self.num_heads)
        )  # (tot_num_heads, num_features_head, num_features // tot_num_heads)

        Wq2 = self.param(
            'Wq2',
            jax.nn.initializers.lecun_normal(batch_axis=(0,)),
            (len(self.degrees), num_features // len(self.degrees), num_features // len(self.degrees))
        )  # (tot_num_heads, num_features_head, num_features // tot_num_heads)

        Wk2 = self.param(
            'Wk2',
            jax.nn.initializers.lecun_normal(batch_axis=(0,)),
            (len(self.degrees), num_features // len(self.degrees), num_features // len(self.degrees))
        )  # (tot_num_heads, num_features_head, num_features // tot_num_heads)

        inv_split1_h, x1_h = split_in_heads(x, num_heads=self.num_heads)
        inv_split2_h, x2_h = split_in_heads(x, num_heads=len(self.degrees))

        q1_i = self.qk_non_linearity(jnp.einsum('Hij, NHj -> NHi', Wq1, x1_h))[idx_i]
        # (num_pairs, tot_num_heads, num_features_head)
        k1_j = self.qk_non_linearity(jnp.einsum('Hij, NHj -> NHi', Wk1, x1_h))[idx_j]
        # (num_pairs, tot_num_heads, num_features_head)

        q2_i = self.qk_non_linearity(jnp.einsum('Hij, NHj -> NHi', Wq2, x2_h))[idx_i]
        # (num_pairs, tot_num_heads, num_features_head)
        k2_j = self.qk_non_linearity(jnp.einsum('Hij, NHj -> NHi', Wk2, x2_h))[idx_j]
        # (num_pairs, tot_num_heads, num_features_head)

        if self.normalization == 'identity':
            nc1 = jnp.array([1.])
            nc2 = jnp.array([1.])
        elif self.normalization == 'sqrt_num_features':
            nc1 = jnp.sqrt(q1_i.shape[-1])
            nc2 = jnp.sqrt(q2_i.shape[-1])
        elif self.normalization == 'avg_num_neighbors':
            nc1 = jnp.asarray(self.avg_num_neighbors)
            nc2 = jnp.asarray(self.avg_num_neighbors)
        else:
            raise ValueError(f'{self.normalization} is not supported.')

        nc1 = nc1.astype(x.dtype)
        nc2 = nc2.astype(x.dtype)

        alpha1_ij = (q1_i * w1_ij * k1_j).sum(axis=-1) / nc1
        alpha1_ij = mask.safe_scale(alpha1_ij, jnp.expand_dims(cut, axis=-1))
        # (num_pairs, num_heads)
        alpha2_ij = (q2_i * w2_ij * k2_j).sum(axis=-1) / nc2
        alpha2_ij = mask.safe_scale(alpha2_ij, jnp.expand_dims(cut, axis=-1))
        # (num_pairs, num_degrees)

        # Aggregation for invariant features
        Wv1 = self.param(
            'Wv1',
            self.value_kernel_init,
            (self.num_heads, num_features // self.num_heads, num_features // self.num_heads)
        )  # (tot_num_heads, num_features // tot_num_heads, num_features // tot_num_heads)

        v_j = jnp.einsum('hij, Nhj -> Nhi', Wv1, x1_h)[idx_j]  # (num_pairs, num_heads, num_features_head)

        x_att = segment_sum(
            jnp.expand_dims(alpha1_ij, axis=-1) * v_j,
            segment_ids=idx_i,
            num_segments=x.shape[0]
        )  # (N, num_heads, num_features_head)

        x_att = inv_split1_h(x_att)  # (N, num_features)
        assert x_att.shape == x.shape

        # Aggregation for Euclidean variables
        ev_att = segment_sum(
            degree_repeat_fn(alpha2_ij) * ylm_ij,
            segment_ids=idx_i,
            num_segments=x.shape[0]
        )  # (N, num_degrees)

        assert ev_att.shape == ev.shape

        return x_att, ev_att


class ExchangeBlock(nn.Module):
    degrees: Sequence[int]
    activation_fn: Callable = jax.nn.silu
    output_is_zero_at_init: bool = False

    def setup(self):
        if self.output_is_zero_at_init:
            self.last_layer_kernel_init = jax.nn.initializers.zeros
        else:
            self.last_layer_kernel_init = jax.nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, x, ev, *args, **kwargs):
        """

        Args:
            x (Array): shape: (N,num_features)
            ev (Array): shape: (N,m_tot)
            *args ():
            **kwargs ():

        Returns:

        """
        num_features = x.shape[-1]
        num_degrees = len(self.degrees)

        contraction_fn = make_l0_contraction_fn(self.degrees, dtype=x.dtype)
        degree_repeat_fn = make_degree_repeat_fn(self.degrees, axis=-1)

        y = jnp.concatenate([x, contraction_fn(ev)], axis=-1)  # shape: (N, num_features+num_degrees)
        # y = self.activation_fn(y)
        # y = nn.Dense(
        #     features=num_features,
        #     name='mlp_layer_1'
        # )(y)  # (N, num_features)
        # y = self.activation_fn(y)
        y = nn.Dense(
            features=num_features + num_degrees,
            kernel_init=self.last_layer_kernel_init,
            name='mlp_layer_2'
        )(y)  # (N, num_features + num_degrees)
        cx, cev = jnp.split(
            y,
            indices_or_sections=np.array([num_features]),
            axis=-1
        )  # (N, num_features) / (N, num_degrees)
        return cx, degree_repeat_fn(cev) * ev


class AttentionBlockDraft(nn.Module):
    """Currently not used."""

    degrees: Sequence[int]
    num_heads: int = 4
    num_features_head: int = 32
    qk_non_linearity: Callable = jax.nn.silu
    activation_fn: Callable = jax.nn.silu
    output_is_zero_at_init: bool = False
    use_spherical_filter: bool = True

    def setup(self):
        if self.output_is_zero_at_init:
            self.value_kernel_init = jax.nn.initializers.zeros
        else:
            self.value_kernel_init = jax.nn.initializers.lecun_normal(batch_axis=(0,))

    @nn.compact
    def __call__(self,
                 x: jnp.ndarray,
                 ev: jnp.ndarray,
                 rbf_ij: jnp.ndarray,
                 ylm_ij: jnp.ndarray,
                 cut: jnp.ndarray,
                 idx_i: jnp.ndarray,
                 idx_j: jnp.ndarray,
                 *args,
                 **kwargs):
        """

        Args:
            x (Array): Node features, shape: (num_nodes, num_features)
            ev (Array): Euclidean variables, shape: (num_nodes, num_orders)
            rbf_ij (Array): RBF expanded distances, shape: (num_pairs, K)
            ylm_ij (Array): Spherical harmonics from i to j, shape: (num_pairs, num_orders)
            cut (Array): Output of the cutoff function feature block, shape: (num_pairs)
            idx_i (Array): index centering atom, shape: (num_pairs)
            idx_j (Array): index neighboring atom, shape: (num_pairs)
            *args ():
            **kwargs ():

        Returns:

        """
        assert x.ndim == 2
        assert ev.ndim == 2
        assert idx_i.ndim == 1
        assert idx_j.ndim == 1
        assert cut.ndim == 1

        num_features = x.shape[-1]
        assert num_features % self.num_heads == 0

        tot_num_heads = self.num_heads + len(self.degrees)
        assert num_features % tot_num_heads == 0

        tot_num_features = tot_num_heads * self.num_features_head

        contraction_fn = make_l0_contraction_fn(self.degrees, dtype=ev.dtype)
        degree_repeat_fn = make_degree_repeat_fn(self.degrees, axis=-1)

        rbf_ij = rbf_ij * jnp.expand_dims(cut, axis=-1)

        w_ij = nn.Dense(
            features=tot_num_features,
            name='radial_filter_layer_2'
        )(
            self.activation_fn(
                nn.Dense(
                    features=tot_num_features // 2,
                    name='radial_filter_layer_1')(rbf_ij)
            )
        )  # (num_pairs, tot_num_features)

        ev_i = ev[idx_i]  # (num_pairs)
        ev_j = ev[idx_j]  # (num_pairs)

        if self.use_spherical_filter:
            w_ij += nn.Dense(
                features=tot_num_features,
                name='spherical_filter_layer_2'
            )(
                self.activation_fn(
                    nn.Dense(
                        features=tot_num_features // 2,
                        name='spherical_filter_layer_1')(contraction_fn(ev_j - ev_i))
                )
            )  # (num_pairs, tot_num_features)

        _, w_ij = split_in_heads(w_ij, num_heads=tot_num_heads)
        # _, (num_pairs, tot_num_heads, num_features_head)

        Wq = self.param(
            'Wq',
            jax.nn.initializers.lecun_normal(batch_axis=(0,)),
            (tot_num_heads, self.num_features_head, num_features // tot_num_heads)
        )  # (tot_num_heads, num_features_head, num_features // tot_num_heads)

        Wk = self.param(
            'Wk',
            jax.nn.initializers.lecun_normal(batch_axis=(0,)),
            (tot_num_heads, self.num_features_head, num_features // tot_num_heads)
        )  # (tot_num_heads, num_features_head, num_features // tot_num_heads)

        inv_split_H, x_H = split_in_heads(x, num_heads=tot_num_heads)

        q_i = self.qk_non_linearity(jnp.einsum('Hij, NHj -> NHi', Wq, x_H))[idx_i]
        # (num_pairs, tot_num_heads, num_features_head)
        k_j = self.qk_non_linearity(jnp.einsum('Hij, NHj -> NHi', Wk, x_H))[idx_j]
        # (num_pairs, tot_num_heads, num_features_head)

        alpha_ij = mask.safe_scale((q_i * w_ij * k_j).sum(axis=-1), jnp.expand_dims(cut, axis=-1))
        # (num_pairs, tot_num_heads)

        alpha1_ij, alpha2_ij = jnp.split(alpha_ij, indices_or_sections=np.array([self.num_heads]), axis=-1)
        # (num_pairs, num_heads), (num_pairs, num_degrees)

        # Aggregation for invariant features
        Wv = self.param(
            'Wv',
            self.value_kernel_init,
            (self.num_heads, num_features // self.num_heads, num_features // self.num_heads)
        )  # (tot_num_heads, num_features // tot_num_heads, num_features // tot_num_heads)

        inv_split_h, x_h = split_in_heads(
            x,
            num_heads=self.num_heads
        )  # fn, (N, num_features // num_heads, num_features_head)

        v_j = jnp.einsum('hij, Nhj -> Nhi', Wv, x_h)[idx_j]  # (num_pairs, num_heads, num_features_head)

        x_att = segment_sum(
            jnp.expand_dims(alpha1_ij, axis=-1) * v_j,
            segment_ids=idx_i,
            num_segments=x.shape[0]
        )  # (N, num_heads, num_features_head)

        x_att = inv_split_h(x_att)  # (N, num_features)
        assert x_att.shape == x.shape

        # Aggregation for Euclidean variables
        ev_att = segment_sum(
            degree_repeat_fn(alpha2_ij) * ylm_ij,
            segment_ids=idx_i,
            num_segments=x.shape[0]
        )  # (N, num_degrees)

        assert ev_att.shape == ev.shape

        return x_att, ev_att