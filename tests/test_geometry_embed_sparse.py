import pytest

import jax
import jax.numpy as jnp

import numpy.testing as npt

from mlff.nn import GeometryEmbedSparse


degrees = [0, 1, 2]

positions = jnp.array([
    [0., 0., 0.],
    [0., -2., 0.],
    [1., 0.5, 0.],
])

idx_i = jnp.array([0, 0, 1, 2])
idx_j = jnp.array([1, 2, 0, 0])

inputs = dict(positions=positions,
              idx_i=idx_i,
              idx_j=idx_j,
              cell=None,
              cell_offset=None)


def test_init():
    geometry_embed = GeometryEmbedSparse(degrees=degrees,
                                         radial_basis_fn='bernstein',
                                         num_radial_basis_fn=16,
                                         cutoff_fn='exponential_cutoff_fn',
                                         cutoff=5.,
                                         input_convention='positions',
                                         prop_keys=None)

    _ = geometry_embed.init(
        jax.random.PRNGKey(0),
        inputs
    )


@pytest.mark.parametrize("cutoff", [1.5, 2.5])
def test_apply(cutoff: float):
    geometry_embed = GeometryEmbedSparse(degrees=degrees,
                                         radial_basis_fn='bernstein',
                                         num_radial_basis_fn=16,
                                         cutoff_fn='exponential_cutoff_fn',
                                         cutoff=cutoff,
                                         input_convention='positions',
                                         prop_keys=None)

    params = geometry_embed.init(
        jax.random.PRNGKey(0),
        inputs
    )

    output = geometry_embed.apply(params, inputs)

    npt.assert_equal(output.get('ylm_ij').shape, (4, 9))
    npt.assert_equal(output.get('rbf_ij').shape, (4, 16))
    npt.assert_allclose(output.get('d_ij'), jnp.array([2., jnp.sqrt(1.25), 2., jnp.sqrt(1.25)]))
    npt.assert_allclose(
        output.get('r_ij'),
        jnp.array(
            [
                [0.0, -2.0, 0.0],
                [1.0, 0.5, 0.0],
                [0.0, 2.0, 0.0],
                [-1.0, -0.5, 0.0],
            ]
        )
    )
    npt.assert_allclose(
        output.get('unit_r_ij'),
        jnp.array(
            [
                [0.0, -1.0, 0.0],
                [1.0/jnp.sqrt(1.25), 0.5/jnp.sqrt(1.25), 0.0],
                [0.0, 1.0, 0.0],
                [-1.0/jnp.sqrt(1.25), -0.5/jnp.sqrt(1.25), 0.0],
            ]
        )
    )

    if cutoff == 2.5:
        npt.assert_allclose(output.get('cut') > 0., jnp.array([True, True, True, True]))
        with npt.assert_raises(AssertionError):
            npt.assert_allclose(output.get('cut') > 0., jnp.array([False, True, False, True]))
    elif cutoff == 1.5:
        npt.assert_allclose(output.get('cut') > 0., jnp.array([False, True, False, True]))
        with npt.assert_raises(AssertionError):
            npt.assert_allclose(output.get('cut') > 0., jnp.array([True, True, True, True]))
    else:
        raise RuntimeError('Invalid test argument.')


