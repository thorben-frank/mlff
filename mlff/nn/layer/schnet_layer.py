import jax.numpy as jnp
import flax.linen as nn
import jax

from typing import (Sequence, Dict, Any)

from mlff.nn.base.sub_module import BaseSubModule
from mlff.masking.mask import safe_scale
from mlff.nn.activation_function.activation_function import shifted_softplus
from mlff.nn.activation_function.activation_function import get_activation_fn
from mlff.nn.mlp import MLP
import logging


class SchNetLayer(BaseSubModule):
    F: int  # feature dimensions for each atom in a sample
    in2f_features: Sequence[int]  # list of dimensions of layers of in2f-MLP
    filter_features: Sequence[int]  # list of dimensions of layers for filter-MLP
    f2out_features: Sequence[int]  # list of dimensions of layers of f2out-MLP
    activation_name: str = "shifted_softplus"  # name of activation function
    module_name: str = 'schnet_layer'  # name of module

    def setup(self):
        self.activation_fn = get_activation_fn(self.activation_name)
        if self.activation_fn is None:
            logging.warning("activation function ", self.activation_name, "not known.")
            self.activation_fn = shifted_softplus

    @nn.compact
    def __call__(self,
                 x: jnp.ndarray,
                 rbf_ij: jnp.ndarray,
                 sph_ij: jnp.ndarray,
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
            sph_ij (Array): Per degree distances of SPHCs, shape: (n_all_pairs,|L|)
            phi_r_cut (Array): Output of the cutoff function feature block, shape: (n_pairs)
            idx_i (Array): index centering atom, shape: (n_pairs)
            idx_j (Array): index neighboring atom, shape: (n_pairs)
            pair_mask (Array): index based mask to exclude pairs that come from index padding, shape: (n_pairs)
            *args ():
            **kwargs ():

        Returns:

        """
        y = MLP(self.in2f_features, use_bias=False, name="in2f")(x)  # shape: (n,in2f_features[-1])
        Wij = MLP(self.filter_features, name="filter_network")(rbf_ij)  # shape: (n_pairs,filter_features[-1])

        # scaling filter weights with atom-pairwise cut-off'ed distances by broadcasting.
        Wij = safe_scale(Wij, phi_r_cut[:, None])  # shape: (n_pairs,filter_features[-1])

        m_ij = safe_scale(y[idx_j] * Wij, pair_mask[:, None])  # shape:(n_pairs,filter_features[-1])
        v = jax.ops.segment_sum(m_ij, idx_i, len(x))  # shape: (n_atoms,filter_features[-1])

        v = MLP(self.f2out_features, name="f2out")(v)  # shape: (n_atoms,F)
        return {'x': x + v}

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {'F': self.F,
                                   'in2f_features': self.in2f_features,
                                   'f2out_features': self.f2out_features,
                                   'filter_features': self.filter_features,
                                   'activation_name': self.activation_name,
                                   'module_name': self.module_name
                                   }
                }


class Sequential(nn.Module):
    layers: Sequence[nn.Module]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    """
    performs the successive calls of list of modules on x 
    Args:
        layers (Array): list of modules that ought to be called successively 

    Returns:
    """
