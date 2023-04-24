# import jax.numpy as jnp
#
# import numpy as np
# from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution, ZeroRotation, Stationary)
# from ase import Atoms
# from ase.units import fs, kB
#
# from mlff import mdx
# from mlff import md
#
# # TODO: re-write as test
# save_dir = '/Users/thorbenfrank/Desktop/'
#
# # use ASE for all preprocessing steps
# data = np.load('/Users/thorbenfrank/Documents/data/mol-data/dft/ethanol_dft.npz')
# atoms = Atoms(positions=data['R'][10], numbers=data['z'])
# ckpt_dir = "/Users/thorbenfrank/Desktop/test/module_2"
#
# seed = 0
# MaxwellBoltzmannDistribution(atoms, temperature_K=300, rng=np.random.default_rng(seed))
# Stationary(atoms)
# ZeroRotation(atoms)
#
# # when it comes to running dynamics switch to mdx
# potential = mdx.MLFFPotential.create_from_ckpt_dir(ckpt_dir=ckpt_dir, dtype=jnp.float32)
# calc = mdx.CalculatorX.create(potential)
# integratorx = mdx.VelocityVerletX.create(timestep=0.5 * fs, calculator=calc)
# simulator = mdx.SimulatorX(n_atoms=len(atoms.get_atomic_numbers()),
#                            save_frequency=1,
#                            run_interval=1,
#                            save_dir=save_dir)
#
# atomsx = mdx.AtomsX.create(atoms=atoms)
# atomsx = atomsx.init_spatial_partitioning(cutoff=potential.cutoff, skin=0.0, capacity_multiplier=1.1)
# simulator.run(integratorx, atomsx, steps=100)
#
# # MD with ASE
# # init a fresh atoms object
# atoms = Atoms(positions=data['R'][10], numbers=data['z'])
# ase_calc = md.mlffCalculator.create_from_ckpt_dir(ckpt_dir=ckpt_dir)
# atoms.set_calculator(ase_calc)
#
# integrator = md.VelocityVerlet(atoms=atoms,
#                                timestep=0.5 * fs,
#                                trajectory=None,
#                                logfile=None,
#                                loginterval=1, )
# simulator = md.Simulator(atoms,
#                          integrator,
#                          300 * kB,
#                          zero_linear_momentum=True,
#                          zero_angular_momentum=True,
#                          save_frequency=1,
#                          save_dir=save_dir,
#                          restart=False,
#                          seed=seed)
#
# simulator.run(100)
#
#
# from ase.io import Trajectory
# import os
# import h5py
#
# traj = Trajectory(os.path.join(save_dir, 'atom.traj'))
# R_ase = [x.get_positions() for x in traj]
# vel_ase = [x.get_velocities() for x in traj]
#
#
# file = h5py.File(os.path.join(save_dir, '../mlff/examples/trajectory.h5'))
# R_mdx = np.array(file['positions'])
# R_mdx = R_mdx.reshape(-1, *R_mdx.shape[2:])
# vel_mdx = np.array(file['velocities'])
# vel_mdx = vel_mdx.reshape(-1, *vel_mdx.shape[2:])
#
# assert np.isclose(R_mdx, np.array(R_ase)[1:], atol=1e-5).all()
# assert np.isclose(vel_mdx, np.array(vel_ase)[1:], atol=1e-5).all()
