import jax.numpy as jnp
import os

from typing import Callable, Type

from flax import struct
from flax.training import checkpoints

from mlff.nn.stacknet import init_stack_net, get_observable_fn
from mlff.io import read_json
from mlff.nn.stacknet import StackNet
from mlff.utils import Graph

from .machine_learning_potential import MachineLearningPotential




def get_model_cutoff(x: StackNet) -> float:
    for g in x.geometry_embeddings:
        if g.module_name == 'geometry_embed':
            model_cut = g.r_cut
            return model_cut
    raise ValueError('No model cutoff defined.')


def get_number_of_steps(x: StackNet) -> int:
    return len(x.layers)


@struct.dataclass
class MLFFPotential(MachineLearningPotential):
    cutoff: float = struct.field(pytree_node=False)
    effective_cutoff: float = struct.field(pytree_node=False)

    potential_fn: Callable[[Graph], jnp.ndarray] = struct.field(pytree_node=False)
    dtype: Type = struct.field(pytree_node=False)  # TODO: remove and determine based on dtype of atomsx

    @classmethod
    def create_from_ckpt_dir(cls, ckpt_dir: str, add_shift: bool = True, dtype=jnp.float32):
        """


        Args:
            ckpt_dir ():
            add_shift ():
            dtype ():

        Returns:

        """

        net = init_stack_net(read_json(os.path.join(ckpt_dir, 'hyperparameters.json')))
        scales = read_json(os.path.join(ckpt_dir, 'scales.json'))
        params = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=None, prefix='checkpoint_loss_')['params']

        _prop_keys = {'displacement_vector': 'displacements',
                      'atomic_energy': net.prop_keys['energy']}

        net.reset_prop_keys(_prop_keys)
        net.reset_input_convention('displacements')
        net.reset_output_convention('per_atom')

        cutoff = get_model_cutoff(net)
        steps = get_number_of_steps(net)

        effective_cutoff = steps * cutoff

        prop_keys = net.prop_keys
        z_key = prop_keys['atomic_type']
        E_key = prop_keys['atomic_energy']

        def scale(v):
            return jnp.asarray(scales['energy']['scale'], dtype=dtype) * v

        def shift(v, z):
            return v + jnp.asarray(scales['energy']['per_atom_shift'], dtype=dtype)[z][:, None]

        if add_shift:
            def scale_and_shift_fn(x: jnp.ndarray, z: jnp.ndarray):
                return shift(scale(x), z)
        else:
            def scale_and_shift_fn(x: jnp.ndarray, z: jnp.ndarray):
                return scale(x)

        obs_fn = get_observable_fn(net)

        def graph_to_mlff_input(graph: Graph):
            idx_i = jnp.where(graph.mask, graph.centers, -1)
            idx_j = jnp.where(graph.mask, graph.others, -1)

            x = {'displacements': graph.edges,
                 z_key: graph.nodes,
                 'idx_i': idx_i,
                 'idx_j': idx_j}

            return x

        def potential_fn(graph: Graph):
            x = graph_to_mlff_input(graph)
            y = obs_fn(params, x)
            return scale_and_shift_fn(y[E_key], x[z_key]).reshape(-1).astype(dtype)

        return cls(cutoff=cutoff,
                   effective_cutoff=effective_cutoff,
                   potential_fn=potential_fn,
                   dtype=dtype)

    def __call__(self, graph: Graph) -> jnp.ndarray:
        return self.potential_fn(graph)
