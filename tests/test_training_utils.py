import jax.numpy as jnp
import numpy.testing as npt
from mlff import training_utils


def test_optimizer():
    optimizer = training_utils.make_optimizer(
        name='adamw',
        learning_rate=1e-3,
        learning_rate_schedule='exponential_decay',
        learning_rate_schedule_args=dict(transition_steps=10, decay_rate=0.7),
        num_of_nans_to_ignore=2
    )

    _ = optimizer.init({'weight': jnp.ones((20, 50))})


def test_node_mse_loss():
    y = jnp.array([1., 0.2, -3., 0.5])
    y_label = jnp.array([0.9, 0.4, -2.7, 0.])

    batch_segments = jnp.array([0, 1, 1, 2])
    graph_mask = jnp.array([True, True, False])

    loss_value = training_utils.node_mse_loss(
        y=y,
        y_label=y_label,
        batch_segments=batch_segments,
        graph_mask=graph_mask,
        scale=jnp.asarray(1.).reshape(-1)
    )
    expected_loss = 1/2 * (1/1 * 0.1**2 + 1/2 * (0.2**2 + 0.3**2))

    npt.assert_allclose(loss_value, expected_loss, atol=1e-6)


def test_node_mse_loss_nan():
    y = jnp.array([1., 0.2, -3., 0.5])
    y_label = jnp.array([0.9, jnp.nan, jnp.nan, 0.])

    batch_segments = jnp.array([0, 1, 1, 2])
    graph_mask = jnp.array([True, True, False])

    loss_value = training_utils.node_mse_loss(
        y=y,
        y_label=y_label,
        batch_segments=batch_segments,
        graph_mask=graph_mask,
        scale=jnp.asarray(1.).reshape(-1)
    )

    expected_loss = 1 / 1 * (1 / 1 * 0.1 ** 2)

    npt.assert_allclose(loss_value, expected_loss, atol=1e-6)
