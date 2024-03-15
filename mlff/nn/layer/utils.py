import jax
import jax.numpy as jnp
import flax.linen as nn
import e3x

from typing import Any, Callable, Optional, Sequence


class Residual(nn.Module):
    """Residual block."""

    num_blocks: int = 2
    activation_fn: Callable[..., Any] = e3x.nn.activations.silu
    use_bias: bool = True
    kernel_init: Callable[..., Any] = jax.nn.initializers.lecun_normal()
    kernel_init_last_block: Callable[..., Any] = jax.nn.initializers.zeros
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs):
        feat = inputs.shape[-1]
        x = inputs

        for i in range(self.num_blocks - 1):
            x = self.activation_fn(x)
            x = e3x.nn.modules.Dense(
                feat,
                use_bias=self.use_bias,
                dtype=self.dtype,
                kernel_init=self.kernel_init,
                param_dtype=self.param_dtype,
                name=f"dense_{i}",
            )(x)

        x = e3x.nn.modules.Dense(
            feat,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            kernel_init=self.kernel_init_last_block,
            name=f"dense_{self.num_blocks - 1}",
        )(x)

        return e3x.nn.add(inputs + x)


class ResidualMLP(nn.Module):
    """Residual MLP."""

    num_residuals: int = 1
    num_blocks_per_residual: int = 2
    use_bias: bool = True
    activation_fn: Callable[..., Any] = e3x.nn.activations.silu
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs):
        feat = inputs.shape[-1]
        x = inputs
        for i in range(self.num_residuals):
            x = Residual(
                self.num_blocks_per_residual,
                self.activation_fn,
                use_bias=self.use_bias,
                dtype=self.dtype,
                param_dtype=self.param_dtype,
                name=f"residual_{i}",
            )(x)

        x = self.activation_fn(x)
        x = e3x.nn.modules.Dense(
            feat,
            use_bias=self.use_bias,
            dtype=self.dtype,
            param_dtype=self.param_dtype,
            name="dense_0",
        )(x)

        return x
