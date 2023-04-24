import jax.numpy as jnp
import flax.linen as nn

from typing import (Any, Dict, Sequence)

from mlff.nn.base.sub_module import BaseSubModule
from mlff.masking.mask import safe_scale
from mlff.nn.mlp import MLP
from mlff.nn.activation_function.activation_function import silu
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
        if self.output_convention == 'per_atom':
            return {self.atomic_energy_key: e_loc[:, None]}
        elif self.output_convention == 'per_structure':
            return {self.energy_key: e_loc.sum(axis=-1, keepdims=True)}
        else:
            raise ValueError(f"{self.output_convention} is invalid argument for attribute `output_convention`.")

    def reset_output_convention(self, output_convention):
        self.output_convention = output_convention

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {'per_atom_scale': self.per_atom_scale,
                                   'per_atom_shift': self.per_atom_shift,
                                   'num_embeddings': self.num_embeddings,
                                   'output_convention': self.output_convention,
                                   'prop_keys': self.prop_keys}
                }
