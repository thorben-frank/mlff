from typing import Dict, Any
from pathlib import Path
import os

from orbax.checkpoint import PyTreeCheckpointer, Checkpointer, PyTreeCheckpointHandler
from orbax import checkpoint
import pathlib

__STEP_PREFIX__: str = 'ckpt'


def load_params_from_ckpt_dir(ckpt_dir):
    loaded_mngr = checkpoint.CheckpointManager(
        pathlib.Path(ckpt_dir).resolve(),
        item_names=('state',),
        item_handlers={'state': checkpoint.StandardCheckpointHandler()},
        options=checkpoint.CheckpointManagerOptions(step_prefix="ckpt"),
    )

    mngr_state = loaded_mngr.restore(
        loaded_mngr.latest_step()
    )

    state = mngr_state.get('state')

    return state['valid_params']


def load_state_from_ckpt_dir(ckpt_dir: str):
    # mngr = CheckpointManager(ckpt_dir, __CHECKPOINTERS__, options=CheckpointManagerOptions(step_prefix=__STEP_PREFIX__))
    # return mngr.restore(n)['state']

    ns = []
    abs_ckpt_dir = Path(ckpt_dir).resolve().absolute()
    for u in os.scandir(abs_ckpt_dir):
        if u.is_dir():
            dir_name = Path(u).stem
            prefix_n = dir_name.split('_')
            if len(prefix_n) == 2:
                prefix, n = prefix_n
                if prefix == __STEP_PREFIX__:
                    ns += [int(n)]
    max_step = max(ns)

    ckptr = Checkpointer(PyTreeCheckpointHandler())
    return ckptr.restore(abs_ckpt_dir / f'{__STEP_PREFIX__}_{max_step}/state', item=None)


def _load_params_from_ckpt_dir(ckpt_dir: str):
    return load_state_from_ckpt_dir(ckpt_dir)['valid_params']
