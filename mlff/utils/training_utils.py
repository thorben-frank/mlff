import jraph
import jax
import jax.numpy as jnp
from mlff.masking.mask import safe_mask
from mlff.utils import jraph_utils
import optax
from typing import Callable, Dict


property_to_mask = {
    'energy': 'graph_mask',
    'forces': 'node_mask',
    'stress': 'graph_mask'
}


def scaled_safe_masked_mse_loss(y, y_true, scale, msk):
    """

    Args:
        y (): shape: (B,d1, *, dN)
        y_true (): (B,d1, *, dN)
        scale (): (d1, *, dN) or everything broadcast-able to (B, d1, *, dN)
        msk (): shape: (B)

    Returns:

    """
    full_mask = ~jnp.isnan(y_true) & msk
    v = safe_mask(full_mask, fn=lambda u: scale * (u - y)**2, operand=y_true)
    den = full_mask.reshape(-1).sum().astype(dtype=v.dtype)
    return safe_mask(den > 0, lambda x: v.reshape(-1).sum() / x, den, 0)


def make_loss_fn(obs_fn: Callable, weights: Dict, scales: Dict = None):
    targets = list(weights.keys())

    if scales is None:
        _scales = {k: jnp.ones(1) for k in targets}
    else:
        _scales = scales

    def loss_fn(params, batch: Dict[str, jnp.ndarray]):
        inputs = {k: v for k, v in batch.items() if k not in targets}
        outputs_true = {k: v for k, v in batch.items() if k in targets}
        outputs_predict = obs_fn(params, **inputs)
        loss = jnp.zeros(1)
        metrics = {}
        for target in targets:
            _l = scaled_safe_masked_mse_loss(y=outputs_predict[target],
                                             y_true=outputs_true[target],
                                             scale=_scales[target],
                                             msk=inputs[property_to_mask[target]]
                                             )

            loss += weights[target] * _l
            metrics.update({target: _l / _scales[target].mean()})

        loss = jnp.reshape(loss, ())
        metrics.update({'loss': loss})

        return loss, metrics

    return loss_fn


def graph_to_batch_fn(graph: jraph.GraphsTuple):
    batch = dict(
        positions=graph.nodes.get('positions'),
        atomic_numbers=graph.nodes.get('atomic_numbers'),
        idx_i=graph.receivers,
        idx_j=graph.senders,
        cell=None,
        cell_offset=None,
        energy=graph.globals.get('energy'),
        forces=graph.nodes.get('forces')
    )
    batch_info = jraph_utils.batch_info_fn(graph)
    batch.update(batch_info)
    return batch


def make_training_step_fn(optimizer, loss_fn):

    @jax.jit
    def training_step_fn(params, opt_state, batch):
        """
        Training step.

        Args:
            params (FrozenDict): Parameter dictionary.
            opt_state: Optax optimizer state.
            batch (Tuple): Batch of validation data.

        Returns:
            Updated state and metrics.

        """
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params, batch)
        updates, opt_state = optimizer.apply_gradients(grads=grads, opt_state=opt_state)
        params = optax.apply_updates(params, updates)
        metrics['gradients_norm'] = optax.global_norm(grads)
        return params, metrics
    return training_step_fn


def make_validation_step_fn(metric_fn):

    @jax.jit
    def validation_step_fn(state, batch) -> Dict[str, jnp.ndarray]:
        """
        Validation step.

        Args:
            state (TrainState): Flax train state.
            batch (Tuple): Batch of validation data.

        Returns:
            Validation metrics.
        """
        _, metrics = metric_fn(state.valid_params, batch)
        return metrics

    return validation_step_fn
