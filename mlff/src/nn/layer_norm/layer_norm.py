import jax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial

from typing import (Callable, Dict, Sequence)

from mlff.src.sph_ops import make_degree_norm_fn
from mlff.src.masking.mask import safe_mask, safe_scale, safe_mask_special


class SPHCLayerNorm(nn.Module):
    degrees: Sequence[int]
    scale_init: Callable = nn.initializers.ones

    def setup(self) -> None:
        self.norm_per_degree_fn = jax.vmap(make_degree_norm_fn(degrees=self.degrees))

        _repeats = [2 * y + 1 for y in self.degrees]
        self.repeat_fn = partial(jnp.repeat, repeats=jnp.array(_repeats), axis=-1, total_repeat_length=sum(_repeats))

    @nn.compact
    def __call__(self, chi: jnp.ndarray, point_mask, *args, **kwargs):
        """

        Args:
            chi (Array): Spherical harmonic coordinates, shape: (n,m_tot)
            *args ():
            **kwargs ():

        Returns:

        """
        scale_weight = self.param('scale_weight', self.scale_init, (len(self.degrees)))  # shape: (|l|)
        #chi_norm = safe_scale(self.norm_per_degree_fn(chi), scale=point_mask[:, None])  # shape: (n,|l|)
        # scale_factor = safe_mask(point_mask[:, None] != 0,
        #                          fn=lambda x: scale_weight / x,
        #                          operand=chi_norm,
        #                          placeholder=0)  # shape: (n,|l|)
        chi_norm = safe_mask(mask=point_mask[:, None] != 0,
                             fn=self.norm_per_degree_fn,
                             operand=chi,
                             placeholder=0)  # shape: (n,|l|)
        chi_norm = self.repeat_fn(chi_norm)  # shape: (n,m_tot)

        chi = safe_mask(mask=chi_norm != 0,
                        fn=lambda y: y / chi_norm,
                        operand=chi,
                        placeholder=0)  # shape: (n,m_tot)

        chi_ = safe_mask(mask=chi != 0,
                         fn=lambda y: y * self.repeat_fn(scale_weight),
                         operand=chi,
                         placeholder=0)  # shape: (n,m_tot)

        return safe_scale(chi_, scale=point_mask[:, None], placeholder=0)  # shape: (n,m_tot)


class LayerNorm(nn.Module):
    use_scale: bool = True
    use_bias: bool = True

    @nn.compact
    def __call__(self, x: jnp.array, *args, **kwargs) -> Dict:
        """

        Args:
            x (Array): Features, shape: (n,F)
            *args ():
            **kwargs ():

        Returns:

        """
        x_norm = nn.LayerNorm(use_scale=self.use_scale, use_bias=self.use_bias)(x)
        return {'x': x_norm}

