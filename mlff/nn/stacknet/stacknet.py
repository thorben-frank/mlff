import flax.linen as nn
import jax.numpy as jnp
import json
import os

from pathlib import Path
from typing import (Any, Callable, Dict, Sequence, Tuple)

from mlff.nn.layer import get_layer
from mlff.nn.embed import get_embedding_module
from mlff.nn.observable import get_observable_module
from mlff.io import read_json


Array = Any


class StackNet(nn.Module):
    geometry_embeddings: Sequence[Callable]
    feature_embeddings: Sequence[Callable]
    layers: Sequence[Callable]
    observables: Sequence[Callable]
    prop_keys: Dict

    def setup(self):
        if len(self.feature_embeddings) == 0:
            msg = "At least one embedding module in `feature_embeddings` is required."
            raise ValueError(msg)
        if len(self.observables) == 0:
            msg = "At least one observable module in `observables` is required."
            raise ValueError(msg)

    @classmethod
    def create_from_ckpt_dir(cls, ckpt_dir: str):
        h_path = Path(ckpt_dir).absolute().resolve() / 'hyperparameters.json'
        stack_net = init_stack_net(read_json(h_path))
        return stack_net

    @nn.compact
    def __call__(self,
                 inputs,
                 *args,
                 **kwargs) -> Dict[str, jnp.ndarray]:
        """
        Energy function of the NN.

        Args:
            inputs (Dict):
            args (Tuple):
            kwargs (Dict):

        Returns: energy, shape: (1)

        """

        quantities = {}
        quantities.update(inputs)

        # Initialize masks
        quantities.update(init_masks(z=inputs[self.prop_keys['atomic_type']],
                                     idx_i=inputs['idx_i'])
                          )

        # Initialize the geometric quantities
        for geom_emb in self.geometry_embeddings:
            geom_quantities = geom_emb(quantities)
            quantities.update(geom_quantities)

        # Initialize the per atom embedding
        embeds = []
        for embed_fn in self.feature_embeddings:
            embeds += [embed_fn(quantities)]  # len: n_embeds, shape: (n,F)
        x = jnp.stack(embeds, axis=-1).sum(axis=-1) / jnp.sqrt(len(embeds))  # shape: (n,F)
        quantities.update({'x': x})

        for (n, layer) in enumerate(self.layers):

            updated_quantities = layer(**quantities)
            quantities.update(updated_quantities)

        observables = {}
        for o_fn in self.observables:
            o_dict = o_fn(quantities)
            observables.update(o_dict)

        # return jax.tree_map(lambda y: y[..., None], observables)
        return observables

    def __dict_repr__(self):
        geometry_embeddings = [x.__dict_repr__() for x in self.geometry_embeddings]
        feature_embeddings = []
        layers = []
        observables = []
        for x in self.feature_embeddings:
            feature_embeddings += [x.__dict_repr__()]
        for (n, x) in enumerate(self.layers):
            layers += [x.__dict_repr__()]
        for x in self.observables:
            observables += [x.__dict_repr__()]

        return {'stack_net': {'geometry_embeddings': geometry_embeddings,
                              'feature_embeddings': feature_embeddings,
                              'layers': layers,
                              'observables': observables,
                              'prop_keys': self.prop_keys,
                              'n_layers': len(layers)}}

    def to_json(self, ckpt_dir, name='hyperparameters.json'):
        j = self.__dict_repr__()
        with open(os.path.join(ckpt_dir, name), 'w', encoding='utf-8') as f:
            json.dump(j, f, ensure_ascii=False, indent=4)

    def reset_prop_keys(self, prop_keys, sub_modules=True) -> None:
        self.prop_keys.update(prop_keys)
        if sub_modules:
            all_modules = self.geometry_embeddings + self.feature_embeddings + self.observables
            for m in all_modules:
                m.reset_prop_keys(prop_keys=prop_keys)

    def reset_input_convention(self, input_convention):
        for g in self.geometry_embeddings:
            g.reset_input_convention(input_convention=input_convention)

    def reset_output_convention(self, output_convention):
        for o in self.observables:
            o.reset_output_convention(output_convention=output_convention)


def init_masks(z, idx_i):
    point_mask = (z != 0).astype(jnp.float32)  # shape: (n)
    pair_mask = (idx_i != -1).astype(jnp.float32)  # shape: (n_pairs)
    return {'point_mask': point_mask, 'pair_mask': pair_mask}


def init_stack_net(h) -> StackNet:
    _h = h['stack_net']
    geom_embs = [get_embedding_module(*tuple(x.items())[0]) for x in _h['geometry_embeddings']]
    feature_embs = [get_embedding_module(*tuple(x.items())[0]) for x in _h['feature_embeddings']]
    lays = [get_layer(*tuple(x.items())[0]) for x in _h['layers']]
    obs = [get_observable_module(*tuple(x.items())[0]) for x in _h['observables']]
    return StackNet(**{'geometry_embeddings': geom_embs,
                       'feature_embeddings': feature_embs,
                       'layers': lays,
                       'observables': obs,
                       'prop_keys': _h['prop_keys']})
