from typing import Dict, Any

from orbax.checkpoint import CheckpointManager, PyTreeCheckpointer

__CHECKPOINTERS__: Dict[str, Any] = {'state': PyTreeCheckpointer()}


def load_state_from_ckpt_dir(ckpt_dir: str):
    mngr = CheckpointManager(ckpt_dir, __CHECKPOINTERS__)
    return mngr.restore(mngr.latest_step())['state']


def load_params_from_ckpt_dir(ckpt_dir: str):
    return load_state_from_ckpt_dir(ckpt_dir)['valid_params']
