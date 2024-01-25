import jax.numpy as jnp
import numpy as np
import logging
from typing import Callable, Type
from flax import struct
from mlff.utils import Graph
from mlff.mdx.potential.machine_learning_potential import MachineLearningPotential
from mlff.nn.stacknet.observable_function_sparse import get_observable_fn_sparse
import pathlib
from ml_collections import config_dict
from mlff.config import from_config
from orbax import checkpoint
import json


def load_hyperparameters(workdir: str):
    with open(pathlib.Path(workdir) / "hyperparameters.json", "r") as fp:
        cfg = json.load(fp)

    cfg = config_dict.ConfigDict(cfg)
    return cfg


def load_model_from_workdir(workdir: str):
    cfg = load_hyperparameters(workdir)

    loaded_mngr = checkpoint.CheckpointManager(
        pathlib.Path(workdir) / "checkpoints",
        {
            "params": checkpoint.PyTreeCheckpointer(),
        },
        options=checkpoint.CheckpointManagerOptions(step_prefix="ckpt"),
    )
    mgr_state = loaded_mngr.restore(
        loaded_mngr.latest_step(),
        {
            "params": checkpoint.PyTreeCheckpointer(),
        })
    params = mgr_state.get("params")

    model = from_config.make_so3krates_sparse_from_config(cfg)

    return model, params


@struct.dataclass
class MLFFPotentialSparse(MachineLearningPotential):
    cutoff: float = struct.field(pytree_node=False)
    effective_cutoff: float = struct.field(pytree_node=False)

    potential_fn: Callable[[Graph], jnp.ndarray] = struct.field(pytree_node=False)
    dtype: Type = struct.field(pytree_node=False)  # TODO: remove and determine based on dtype of atomsx

    @classmethod
    def create_from_ckpt_dir(cls, ckpt_dir: str, add_shift: bool = False, dtype=jnp.float32):
        logging.warning(
            '`create_from_ckpt_dir` is deprecated and replaced by `create_from_workdir`, please use this method in '
            'the future. For now this calls `create_from_workdir` but will raise an error in the future.'
        )
        return cls.create_from_workdir(ckpt_dir, add_shift, dtype)

    @classmethod
    def create_from_workdir(cls, workdir: str, add_shift: bool = False, dtype=jnp.float32):
        """


        Args:
            workdir ():
            add_shift ():
            dtype ():

        Returns:

        """

        if add_shift is True and (dtype == np.float32 or dtype == jnp.float32):
            import logging
            logging.warning('Energy shift is enabled but float32 precision is used.'
                            ' For large absolute energy values, this can lead to floating point errors in the energies.'
                            ' If you do not need the absolute energy values since only relative ones are important, we'
                            ' suggest to disable the energy shift since increasing the precision slows down'
                            ' computation.')

        net, params = load_model_from_workdir(workdir=workdir)
        cfg = load_hyperparameters(workdir=workdir)

        net.reset_input_convention('displacements')
        net.reset_output_convention('per_atom')

        cutoff = cfg.model.cutoff
        steps = cfg.model.num_layers

        effective_cutoff = steps * cutoff

        if add_shift:
            shifts = {int(k): float(v) for k, v in dict(cfg.data.energy_shifts).items()}

            def shift(v, z):
                return v + jnp.asarray(shifts, dtype=dtype)[z][:, None]
        else:
            def shift(v, z):
                return v

        def shift_fn(x: jnp.ndarray, z: jnp.ndarray):
            return shift(x, z)

        obs_fn = get_observable_fn_sparse(net)

        def graph_to_mlff_input(graph: Graph):

            x = {
                'positions': None,
                'displacements': graph.edges,
                'atomic_numbers': graph.nodes,
                'idx_i': graph.centers,
                'idx_j': graph.others
            }

            return x

        def potential_fn(graph: Graph):
            x = graph_to_mlff_input(graph)
            y = obs_fn(params, **x)
            return shift_fn(y['energy'], x['atomic_numbers']).reshape(-1).astype(dtype)

        return cls(cutoff=cutoff,
                   effective_cutoff=effective_cutoff,
                   potential_fn=potential_fn,
                   dtype=dtype)

    def __call__(self, graph: Graph) -> jnp.ndarray:
        return self.potential_fn(graph)