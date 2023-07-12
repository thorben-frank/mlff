import jax.numpy as jnp

import numpy as np
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution, ZeroRotation, Stationary)
from ase import Atoms
from ase.units import fs, kB

from mlff import mdx

ps = 1000*fs

data = np.load('example_data/ethanol.npz')
atoms = Atoms(positions=data['R'][999], numbers=data['z'])

T0 = 500*kB

MaxwellBoltzmannDistribution(atoms, temperature_K=T0 / kB)

vel = atoms.get_velocities()
T = atoms.get_temperature()

# when it comes to running dynamics switch to mdx
ckpt_dir = "ckpt_dir/module"
dtype = jnp.float32
potential = mdx.MLFFPotential.create_from_ckpt_dir(ckpt_dir=ckpt_dir, add_shift=True, dtype=dtype)
calc = mdx.CalculatorX.create(potential)
integratorx = mdx.VelocityVerletX.create(timestep=0.5*fs, calculator=calc)
simulator = mdx.SimulatorX(n_atoms=len(atoms.get_atomic_numbers()), save_frequency=1, run_interval=1)

atomsx = mdx.AtomsX.create(atoms=atoms, dtype=dtype)
atomsx = atomsx.init_spatial_partitioning(cutoff=potential.cutoff, skin=0.0, capacity_multiplier=1.1)

optimizer = mdx.LBFGS.create(potential=potential, save_dir=None)
atomsx_opt, grads = optimizer.minimize(atomsx, max_steps=200, tol=1e-4)
atomsx_opt = mdx.scale_momenta(atomsx_opt, T0=T0)
atomsx_opt = mdx.zero_rotation(mdx.zero_translation(atomsx_opt))

simulator.run(integratorx, atomsx_opt, steps=200_000)
