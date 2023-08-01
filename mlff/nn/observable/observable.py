import jax.numpy as jnp
import flax.linen as nn

from jax.ops import segment_sum
from jax.nn.initializers import constant

from typing import (Any, Dict, Sequence)

from mlff.nn.base.sub_module import BaseSubModule
from mlff.masking.mask import safe_scale, safe_mask
from mlff.nn.mlp import MLP
from mlff.nn.activation_function.activation_function import silu, softplus_inverse, softplus
import mlff.properties.property_names as pn
Array = Any


def get_observable_module(name, h):
    if name == 'energy':
        return Energy(**h)
    else:
        msg = "No observable module implemented for `module_name={}`".format(name)
        raise ValueError(msg)


class Energy(BaseSubModule):
    prop_keys: Dict
    per_atom_scale: Sequence[float] = None
    per_atom_shift: Sequence[float] = None
    num_embeddings: int = 100
    zbl_repulsion: bool = False
    zbl_repulsion_shift: float = 0.
    output_convention: str = 'per_structure'
    module_name: str = 'energy'

    def setup(self):
        self.energy_key = self.prop_keys[pn.energy]
        self.atomic_type_key = self.prop_keys[pn.atomic_type]
        if self.output_convention == 'per_atom':
            self.atomic_energy_key = self.prop_keys[pn.atomic_energy]

        if self.per_atom_scale is not None:
            self.get_per_atom_scale = lambda y, *args, **kwargs: jnp.take(jnp.array(self.per_atom_scale), y)
            # returns array, shape: (n)
        else:
            self.get_per_atom_scale = lambda *args, **kwargs: jnp.float32(1.)

        if self.per_atom_shift is not None:
            self.get_per_atom_shift = lambda y, *args, **kwargs: jnp.take(jnp.array(self.per_atom_shift), y)
        else:
            self.get_per_atom_shift = lambda *args, **kwargs: jnp.float32(0.)

    @nn.compact
    def __call__(self, inputs: Dict, *args, **kwargs):
        """

        Args:
            inputs ():
            *args ():
            **kwargs ():

        Returns:

        """
        x = inputs['x']
        point_mask = inputs['point_mask']
        z = inputs[self.atomic_type_key].astype(jnp.int16)

        e_loc = MLP(features=[x.shape[-1], 1], activation_fn=silu)(x).squeeze(axis=-1)  # shape: (n)
        e_loc = self.get_per_atom_scale(z) * e_loc + self.get_per_atom_shift(z)  # shape: (n)
        e_loc = safe_scale(e_loc, scale=point_mask)  # shape: (n)

        if self.zbl_repulsion:
            e_rep = ZBLRepulsion(prop_keys=self.prop_keys,
                                 output_convention=self.output_convention)(inputs)  # shape: (n,1) or (1)
            e_rep = e_rep - jnp.asarray(self.zbl_repulsion_shift, dtype=e_rep.dtype)
        else:
            e_rep = jnp.asarray(0., dtype=e_loc.dtype)  # shape: (1)

        if self.output_convention == 'per_atom':
            return {self.atomic_energy_key: e_loc[:, None] + e_rep}
        elif self.output_convention == 'per_structure':
            return {self.energy_key: e_loc.sum(axis=-1, keepdims=True) + e_rep}
        else:
            raise ValueError(f"{self.output_convention} is invalid argument for attribute `output_convention`.")

    def reset_output_convention(self, output_convention):
        self.output_convention = output_convention
        self.zbl_repulsion_energy.reset_output_convention(output_convention)

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {'per_atom_scale': self.per_atom_scale,
                                   'per_atom_shift': self.per_atom_shift,
                                   'num_embeddings': self.num_embeddings,
                                   'output_convention': self.output_convention,
                                   'zbl_repulsion': self.zbl_repulsion,
                                   'zbl_repulsion_shift': self.zbl_repulsion_shift,
                                   'prop_keys': self.prop_keys}
                }


class ZBLRepulsion(nn.Module):
    """
    Ziegler-Biersack-Littmark repulsion.
    """
    prop_keys: dict
    output_convention: str = 'per_structure'
    module_name: str = 'energy_repulsion'
    a0: float = 0.5291772105638411
    ke: float = 14.399645351950548

    def setup(self):
        self.energy_key = self.prop_keys[pn.energy]
        self.atomic_type_key = self.prop_keys[pn.atomic_type]
        self.atomic_position_key = self.prop_keys[pn.atomic_position]

        if self.output_convention == 'per_atom':
            self.atomic_energy_key = self.prop_keys[pn.atomic_energy]

    def reset_output_convention(self, output_convention):
        self.output_convention = output_convention

    @nn.compact
    def __call__(self, inputs: Dict, *args, **kwargs):
        a1 = softplus(self.param('a1', constant(softplus_inverse(3.20000)), (1,)))  # shape: (1)
        a2 = softplus(self.param('a2', constant(softplus_inverse(0.94230)), (1,)))  # shape: (1)
        a3 = softplus(self.param('a3', constant(softplus_inverse(0.40280)), (1,)))  # shape: (1)
        a4 = softplus(self.param('a4', constant(softplus_inverse(0.20160)), (1,)))  # shape: (1)
        c1 = softplus(self.param('c1', constant(softplus_inverse(0.18180)), (1,)))  # shape: (1)
        c2 = softplus(self.param('c2', constant(softplus_inverse(0.50990)), (1,)))  # shape: (1)
        c3 = softplus(self.param('c3', constant(softplus_inverse(0.28020)), (1,)))  # shape: (1)
        c4 = softplus(self.param('c4', constant(softplus_inverse(0.02817)), (1,)))  # shape: (1)
        p = softplus(self.param('p', constant(softplus_inverse(0.23)), (1,)))  # shape: (1)
        d = softplus(self.param('d', constant(softplus_inverse(1 / (0.8854 * self.a0))), (1,)))  # shape: (1)

        c_sum = c1 + c2 + c3 + c4
        c1 = c1 / c_sum
        c2 = c2 / c_sum
        c3 = c3 / c_sum
        c4 = c4 / c_sum

        pair_mask = inputs['pair_mask']

        phi_r_cut_ij = inputs['phi_r_cut']

        d_ij = inputs['d_ij']  # shape: (P)
        z = inputs[self.atomic_type_key]  # shape: (n)
        zf = z.astype(d_ij.dtype)  # shape: (n)

        idx_i = inputs[self.prop_keys[pn.idx_i]]  # shape: (P)
        idx_j = inputs[self.prop_keys[pn.idx_j]]  # shape: (P)
        z_i = zf[idx_i]
        z_j = zf[idx_j]

        z_d_ij = safe_mask(mask=d_ij != 0,
                           operand=d_ij,
                           fn=lambda u: z_i * z_j / u,
                           placeholder=0.
                           )  # shape: (P)

        x = self.ke * phi_r_cut_ij * z_d_ij  # shape: (P)

        rzd = d_ij * (jnp.power(z_i, p) + jnp.power(z_j, p)) / d  # shape: (P)
        y = c1 * jnp.exp(-a1 * rzd) + c2 * jnp.exp(-a2 * rzd) + c3 * jnp.exp(-a3 * rzd) + c4 * jnp.exp(-a4 * rzd)
        # shape: (P)

        e_rep_edge = safe_scale(x * y, scale=pair_mask) / jnp.asarray(2, dtype=d_ij.dtype)  # shape: (P)

        if self.output_convention == 'per_atom':
            return segment_sum(e_rep_edge, segment_ids=idx_i, num_segments=len(z))[:, None]  # shape: (n,1)
        elif self.output_convention == 'per_structure':
            return e_rep_edge.sum(axis=0)  # shape: (1)
        else:
            raise ValueError(f"{self.output_convention} is invalid argument for attribute `output_convention`.")


def make_e_rep_fn(prop_keys, ds, cutoff_fn='cosine_cutoff_fn', r_cut=5, mic=False):
    from mlff.masking.mask import safe_scale
    from mlff.cutoff_function.pbc import add_cell_offsets
    from mlff.cutoff_function import get_cutoff_fn
    import jax

    cut_fn = get_cutoff_fn(cutoff_fn)
    P = ds[0][prop_keys[pn.idx_i]].shape[-1]
    # n = ds[0][prop_keys[pn.atomic_type]].shape[-1]

    init_inputs = {'phi_r_cut': jnp.ones(P),
                   'd_ij': jnp.ones(P),
                   'pair_mask': jnp.zeros(P),
                   **jax.tree_map(lambda x: jnp.array(x[0, ...]), ds[0])}

    zbl_repulsion = ZBLRepulsion(prop_keys=prop_keys)
    p = zbl_repulsion.init(jax.random.PRNGKey(0), init_inputs)
    zbl_repulsion_fn = lambda u: zbl_repulsion.apply(p, u)

    def e_rep_fn(inputs):
        idx_i = inputs['idx_i']  # shape: (n_pairs)
        idx_j = inputs['idx_j']  # shape: (n_pairs)
        pair_mask = (idx_i > -1).astype(jnp.float32)  # shape: (n_pairs)

        R = inputs[prop_keys[pn.atomic_position]]  # shape: (n,3)
        # Calculate pairwise distance vectors
        r_ij = safe_scale(jax.vmap(lambda i, j: R[j] - R[i])(idx_i, idx_j), scale=pair_mask[:, None])
        # shape: (n_pairs,3)

        # Apply minimal image convention if needed
        if mic:
            cell = inputs[prop_keys[pn.unit_cell]]  # shape: (3,3)
            cell_offsets = inputs[prop_keys[pn.cell_offset]]  # shape: (n_pairs,3)
            r_ij = add_cell_offsets(r_ij=r_ij, cell=cell, cell_offsets=cell_offsets)  # shape: (n_pairs,3)

        # Scale pairwise distance vectors with pairwise mask
        r_ij = safe_scale(r_ij, scale=pair_mask[:, None])

        # Calculate pairwise distances
        d_ij = safe_scale(jnp.linalg.norm(r_ij, axis=-1), scale=pair_mask)  # shape : (n_pairs)

        phi_r_cut = safe_scale(cut_fn(d_ij, r_cut=r_cut), scale=pair_mask)  # shape: (n_pairs)

        y = {'phi_r_cut': phi_r_cut, 'd_ij': d_ij, 'pair_mask': pair_mask, **inputs}

        return zbl_repulsion_fn(y)

    return e_rep_fn


def estimate_zbl_repulsion_contribution(prop_keys, ds, cutoff_fn, r_cut, mic=False):
    import jax
    import numpy as np

    batch_size = 100
    N_data = len(ds[0][prop_keys[pn.atomic_position]])
    batch_size = batch_size if batch_size < N_data else N_data
    B = N_data // batch_size
    e_rep_fn = make_e_rep_fn(prop_keys=prop_keys, ds=ds, cutoff_fn=cutoff_fn, r_cut=r_cut, mic=mic)
    e_rep_fn = jax.jit(jax.vmap(e_rep_fn))

    x = np.zeros(batch_size)
    for b in range(B):
        train_batch = jax.tree_map(lambda y: y[int(b*batch_size):int((b+1)*batch_size), ...], ds[0])
        x += np.array(e_rep_fn(train_batch), dtype=np.float64)

    x = x / B
    x = x.mean()
    return x
