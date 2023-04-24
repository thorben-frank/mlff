from abc import abstractmethod
from flax import struct

import jax.numpy as jnp

from typing import Callable, Type
from mlff.utils import Graph


@struct.dataclass
class MachineLearningPotential:
    cutoff: float = struct.field(pytree_node=False)
    effective_cutoff: float = struct.field(pytree_node=False)

    potential_fn: Callable[[Graph], jnp.ndarray] = struct.field(pytree_node=False)
    dtype: Type = struct.field(pytree_node=False)

    @classmethod
    @abstractmethod
    def create_from_ckpt_dir(cls, *args, **kwargs):
        pass

    @abstractmethod
    def __call__(self, graph: Graph):
        pass
