import jax.numpy as jnp
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
