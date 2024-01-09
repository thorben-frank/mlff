from .test_data import load_data


def test_zero_translation():
    import jax.numpy as jnp
    import numpy as np
    from ase import Atoms
    from ase.units import kB
    from ase.md.velocitydistribution import Stationary, MaxwellBoltzmannDistribution

    from mlff import mdx

    data = load_data('ethanol_dep.npz')
    atoms = Atoms(positions=data['R'][100], numbers=data['z'])
    MaxwellBoltzmannDistribution(atoms, temperature_K=500)

    atomsx = mdx.AtomsX.create(atoms, dtype=jnp.float64)

    assert np.isclose(atomsx.get_momenta(), atoms.get_momenta(), atol=1e-5).all()

    Stationary(atoms)

    assert ~np.isclose(atomsx.get_momenta(), atoms.get_momenta(), atol=1e-5).all()

    atomsx = mdx.zero_translation(atomsx)

    # assert np.isclose(atomsx.get_temperature() / jnp.asarray(kB), atoms.get_temperature(), atol=1e-5).all()
    assert np.isclose(atomsx.get_momenta(), atoms.get_momenta(), atol=1e-5).all()
    assert np.isclose(atomsx.get_center_of_mass_velocity(), 0., atol=1e-5).all()


def test_zero_rotation():
    import jax.numpy as jnp
    import numpy as np
    from ase import Atoms
    from ase.units import kB
    from ase.md.velocitydistribution import ZeroRotation, MaxwellBoltzmannDistribution

    from mlff import mdx

    data = load_data('ethanol_dep.npz')
    atoms = Atoms(positions=data['R'][100], numbers=data['z'])
    MaxwellBoltzmannDistribution(atoms, temperature_K=500)

    atomsx = mdx.AtomsX.create(atoms, dtype=jnp.float64)

    assert np.isclose(atomsx.get_angular_momentum(), atoms.get_angular_momentum(), atol=1e-5).all()

    ZeroRotation(atoms)

    assert ~np.isclose(atomsx.get_angular_momentum(), atoms.get_angular_momentum(), atol=1e-5).all()

    atomsx = mdx.zero_rotation(atomsx)

    # assert np.isclose(atomsx.get_temperature() / jnp.asarray(kB), atoms.get_temperature(), atol=1e-5).all()
    assert np.isclose(atomsx.get_angular_momentum(), atoms.get_angular_momentum(), atol=1e-5).all()
    assert np.isclose(atomsx.get_angular_momentum(), 0., atol=1e-5).all()


def test_zero_rotation_and_zero_translation():
    import jax.numpy as jnp
    import numpy as np
    from ase import Atoms
    from ase.units import kB
    from ase.md.velocitydistribution import Stationary, ZeroRotation, MaxwellBoltzmannDistribution

    from mlff import mdx

    data = load_data('ethanol_dep.npz')
    atoms = Atoms(positions=data['R'][100], numbers=data['z'])
    MaxwellBoltzmannDistribution(atoms, temperature_K=500)

    atomsx = mdx.AtomsX.create(atoms, dtype=jnp.float64)

    assert np.isclose(atomsx.get_angular_momentum(), atoms.get_angular_momentum(), atol=1e-5).all()
    assert np.isclose(atomsx.get_momenta(), atoms.get_momenta(), atol=1e-5).all()

    Stationary(atoms)
    ZeroRotation(atoms)

    assert ~np.isclose(atomsx.get_momenta(), atoms.get_momenta(), atol=1e-5).all()
    assert ~np.isclose(atomsx.get_angular_momentum(), atoms.get_angular_momentum(), atol=1e-5).all()

    atomsx = mdx.zero_translation(atomsx)
    atomsx = mdx.zero_rotation(atomsx)

    # assert np.isclose(atomsx.get_temperature() / jnp.asarray(kB), atoms.get_temperature(), atol=1e-5).all()
    assert np.isclose(atomsx.get_angular_momentum(), atoms.get_angular_momentum(), atol=1e-5).all()
    assert np.isclose(atomsx.get_angular_momentum(), 0., atol=1e-5).all()
    assert np.isclose(atomsx.get_momenta(), atoms.get_momenta(), atol=1e-5).all()
    assert np.isclose(atomsx.get_center_of_mass_velocity(), 0., atol=1e-5).all()


def test_momenta_scaling():
    import jax.numpy as jnp
    import numpy as np
    from ase import Atoms
    from ase.units import kB
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    from mlff import mdx

    T0 = 500

    data = load_data('ethanol_dep.npz')
    atoms = Atoms(positions=data['R'][100], numbers=data['z'])
    MaxwellBoltzmannDistribution(atoms, temperature_K=T0)

    atomsx = mdx.AtomsX.create(atoms, dtype=jnp.float64)
    atomsx = mdx.scale_momenta(atomsx, T0=T0 * kB)
    assert np.isclose(atomsx.get_temperature() / kB, T0, atol=1e-5)


# def test_root_mean_square_deviation():
#     from MDAnalysis.analysis import rms
#     from mlff import mdx
#
#     data = dict(load_data('ethanol.npz'))
#
#     positions0 = data['R'][1_263]
#     positions1 = data['R'][1_601]
#
#     mda_rmsd = rms.rmsd(positions0,  # coordinates to align
#                         positions1,  # reference coordinates
#                         center=True,  # subtract the center of geometry
#                         superposition=True)  # superimpose coordinates
#
#     mdx_rmsd = mdx.root_mean_square_deviation(pos0=positions0, pos1=positions1)
