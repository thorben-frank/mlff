import jax.numpy as jnp
import flax.linen as nn

from typing import (Any, Dict, Sequence)

from mlff.src.nn.base.sub_module import BaseSubModule
from mlff.src.masking.mask import safe_scale
from mlff.src.nn.mlp import MLP
from mlff.src.nn.activation_function.activation_function import silu

Array = Any


def get_observable_module(name, h):
    if name == 'energy':
        return Energy(**h)
    else:
        msg = "No observable module implemented for `module_name={}`".format(name)
        raise ValueError(msg)


class Energy(BaseSubModule):
    prop_keys: Dict
    per_atom_scale: Sequence[float]
    per_atom_shift: Sequence[float]
    num_embeddings: int = 100
    module_name: str = 'energy'

    def setup(self):
        self.energy_key = self.prop_keys.get('energy')
        self.atomic_type_key = self.prop_keys.get('atomic_type')

        if self.per_atom_scale is not None:
            self.get_per_atom_scale = lambda y, *args, **kwargs: jnp.take(jnp.array(self.per_atom_scale), y)
        else:
            self.get_per_atom_scale = lambda y, *args, **kwargs: nn.Embed(num_embeddings=self.num_embeddings,
                                                                          features=1)(y).squeeze(axis=-1)
        # returns array, shape: (n)

        if self.per_atom_shift is not None:
            self.get_per_atom_shift = lambda y, *args, **kwargs: jnp.take(jnp.array(self.per_atom_shift), y)
        else:
            self.get_per_atom_shift = lambda y, *args, **kwargs: nn.Embed(num_embeddings=self.num_embeddings,
                                                                          features=1)(y).squeeze(axis=-1)
        # returns array, shape: (n)

    @nn.compact
    def __call__(self, inputs, *args, **kwargs):
        """

        Args:
            inputs ():
            *args ():
            **kwargs ():

        Returns:

        """
        x = inputs['x']
        point_mask = inputs['point_mask']
        z = inputs[self.atomic_type_key]

        e_loc = MLP(features=[x.shape[-1], 1], activation_fn=silu)(x).squeeze(axis=-1)  # shape: (n)
        e_loc = self.get_per_atom_scale(z) * e_loc + self.get_per_atom_shift(z)  # shape: (n)
        e_loc = safe_scale(e_loc, scale=point_mask)  # shape: (n)
        return {self.energy_key: e_loc.sum(axis=-1, keepdims=True)}  # shape: (1)

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {'per_atom_scale': self.per_atom_scale,
                                   'per_atom_shift': self.per_atom_shift,
                                   'num_embeddings': self.num_embeddings,
                                   'prop_keys': self.prop_keys}
                }
