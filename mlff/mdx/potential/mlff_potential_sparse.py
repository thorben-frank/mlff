import jax
import jax.numpy as jnp
import numpy as np
import logging
from typing import Any, Callable, Type, Dict
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


def load_model_from_workdir(
        workdir: str,
        model='so3krates',
        long_range_kwargs: Dict[str, Any] = None,
        from_file: bool = False
):
    cfg = load_hyperparameters(workdir)

    dispersion_energy_bool = cfg.model.dispersion_energy_bool
    electrostatic_energy_bool = cfg.model.electrostatic_energy_bool

    # For local model both are false.
    if (electrostatic_energy_bool is True) or (dispersion_energy_bool is True):
        if long_range_kwargs is None:
            raise ValueError(
                "For a potential with long-range electrostatic and/or dispersion corrections, long_range_kwargs must "
                f"be specified. Received {long_range_kwargs=}."
            )

        cutoff_lr = long_range_kwargs['cutoff_lr']
        neighborlist_format = long_range_kwargs['neighborlist_format_lr']
        if cutoff_lr is not None:
            if cutoff_lr < 0:
                raise ValueError(
                    f"For a potential with long range components the long range cutoff value must be greater "
                    f"than zero. received {cutoff_lr=}."
                )

        cfg.model.cutoff_lr = cutoff_lr
        cfg.neighborlist_format_lr = neighborlist_format

        if dispersion_energy_bool is True:
            dispersion_energy_cutoff_lr_damping = long_range_kwargs['dispersion_energy_cutoff_lr_damping']
            if cutoff_lr is not None:
                if dispersion_energy_cutoff_lr_damping is None:
                    raise ValueError(
                        f"dispersion_energy_cutoff_lr_damping must not be None if dispersion_energy_bool is True and "
                        f"cutoff_lr has a finite value. received {dispersion_energy_bool=}, {cutoff_lr=} and "
                        f"{dispersion_energy_cutoff_lr_damping=}."
                    )
            if cutoff_lr is None:
                if dispersion_energy_cutoff_lr_damping is not None:
                    raise ValueError(
                        f"dispersion_energy_cutoff_lr_damping must be None if dispersion_energy_bool is True and "
                        f"cutoff_lr is infinite (specified via lr_cutoff=None). received {dispersion_energy_bool=}, "
                        f"{dispersion_energy_cutoff_lr_damping=} and {cutoff_lr=}"
                    )
            cfg.model.dispersion_energy_cutoff_lr_damping = dispersion_energy_cutoff_lr_damping

    if from_file is True:
        import pickle

        with open(pathlib.Path(workdir) / 'params.pkl', 'rb') as f:
            params = pickle.load(f)
    else:
        loaded_mngr = checkpoint.CheckpointManager(
            pathlib.Path(workdir) / "checkpoints",
            item_names=('params',),
            item_handlers={'params': checkpoint.StandardCheckpointHandler()},
            options=checkpoint.CheckpointManagerOptions(step_prefix="ckpt"),
        )

        mngr_state = loaded_mngr.restore(
            loaded_mngr.latest_step()
        )

        params = mngr_state.get('params')

    if model == 'so3krates':
        net = from_config.make_so3krates_sparse_from_config(cfg)
    elif model == 'itp_net':
        net = from_config.make_itp_net_from_config(cfg)
    else:
        raise ValueError(
            f'{model=} is not a valid model.'
        )

    return net, params


@struct.dataclass
class MLFFPotentialSparse(MachineLearningPotential):
    cutoff: float = struct.field(pytree_node=False)
    effective_cutoff: float = struct.field(pytree_node=False)

    long_range_bool: bool = struct.field(pytree_node=False)
    long_range_cutoff: float = struct.field(pytree_node=False)

    potential_fn: Callable[[Graph, bool], jnp.ndarray] = struct.field(pytree_node=False)
    dtype: Type = struct.field(pytree_node=False)  # TODO: remove and determine based on dtype of atomsx

    @classmethod
    def create_from_ckpt_dir(
            cls,
            ckpt_dir: str,
            from_file: bool = False,
            add_shift: bool = False,
            long_range_kwargs: Dict[str, Any] = None,
            dtype=jnp.float32,
            model: str = 'so3krates',
    ):
        logging.warning(
            '`create_from_ckpt_dir` is deprecated and replaced by `create_from_workdir`, please use this method in '
            'the future. For now this calls `create_from_workdir` but will raise an error in the future.'
        )
        return cls.create_from_workdir(
            ckpt_dir,
            from_file,
            add_shift,
            long_range_kwargs,
            dtype,
            model,
        )

    @classmethod
    def create_from_workdir(
            cls,
            workdir: str,
            from_file: bool = False,
            add_shift: bool = False,
            long_range_kwargs: Dict[str, Any] = None,
            dtype=jnp.float32,
            model: str = 'so3krates',
    ):
        """


        Args:
            workdir ():
            from_file ():
            add_shift ():
            long_range_kwargs (): Dictionary with keyword arguments for the long-range modules.
            dtype ():
            model ():
            has_aux (): If the potential returns the full output of the model in addition to the atomic energies.

        Returns:

        """

        if add_shift is True and (dtype == np.float32 or dtype == jnp.float32):
            import logging
            logging.warning('Energy shift is enabled but float32 precision is used.'
                            ' For large absolute energy values, this can lead to floating point errors in the energies.'
                            ' If you do not need the absolute energy values since only relative ones are important, we'
                            ' suggest to disable the energy shift since increasing the precision slows down'
                            ' computation.')

        net, params = load_model_from_workdir(
            workdir=workdir,
            from_file=from_file,
            model=model,
            long_range_kwargs=long_range_kwargs
        )

        cfg = load_hyperparameters(workdir=workdir)

        net.reset_input_convention('displacements')
        net.reset_output_convention('per_atom')

        long_range_bool = (cfg.model.electrostatic_energy_bool is True) or (cfg.model.dispersion_energy_bool is True)

        cutoff = cfg.model.cutoff
        # ITPNet is strictly local so has effectively "one" MP layer in terms of effective cutoff.
        steps = cfg.model.num_layers if model != 'itp_net' else 1

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
            # Local case corresponds to no having idx_lr_i, idx_lr_j, displacements_lr = None, None, None
            # TODO: make num_unpaired_electrons, total charge optional maybe using getattr(., ., None)
            #  as this would ensure compatibility with Graph as used in glp.
            x = {
                'positions': graph.positions,
                'displacements': graph.edges,
                'atomic_numbers': graph.nodes,
                'idx_i': graph.centers,
                'idx_j': graph.others,
                'total_charge': graph.total_charge,
                'num_unpaired_electrons': graph.num_unpaired_electrons,
                'cell': getattr(graph, 'cell', None),
            }
            if long_range_bool is True:
                x_lr = {
                    'displacements_lr': graph.edges_lr,
                    'idx_i_lr': graph.idx_i_lr,
                    'idx_j_lr': graph.idx_j_lr,
                }

                x.update(x_lr)

            return x

        def potential_fn(graph: Graph, has_aux: bool = False):
            x = graph_to_mlff_input(graph)
            y = obs_fn(params, **x)
            if has_aux:
                return shift_fn(y['energy'], x['atomic_numbers']).reshape(-1).astype(dtype), jax.tree_map(lambda u: u.astype(dtype), y)
            else:
                return shift_fn(y['energy'], x['atomic_numbers']).reshape(-1).astype(dtype)

        return cls(cutoff=cutoff,
                   effective_cutoff=effective_cutoff,
                   long_range_bool=long_range_bool,
                   long_range_cutoff=long_range_kwargs['cutoff_lr'] if long_range_bool is True else None,
                   potential_fn=potential_fn,
                   dtype=dtype)

    def __call__(self, graph: Graph, has_aux: bool = False) -> jnp.ndarray:
        return self.potential_fn(graph, has_aux)
