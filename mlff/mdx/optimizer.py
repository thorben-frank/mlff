import jax
import jax.numpy as jnp
import jaxopt

import numpy as np
import os

from flax import struct
from typing import Dict, Callable

from .potential.machine_learning_potential import MachineLearningPotential
from .atoms import AtomsX
from .hdfdict import HDF5Store, DataSetEntry


@struct.dataclass
class GradientDescent:
    learning_rate: float = struct.field(pytree_node=False)
    potential: MachineLearningPotential = struct.field(pytree_node=False)

    @classmethod
    def create(cls,
               potential: MachineLearningPotential,
               learning_rate: float = 1e-2):

        return GradientDescent(potential=potential,
                               learning_rate=learning_rate)

    @jax.jit
    def step(self, atoms: AtomsX) -> (AtomsX, jnp.ndarray):
        def energy_fn(y: AtomsX):
            graph = y.to_graph()
            return self.potential(graph).sum()

        grads = jax.grad(energy_fn, allow_int=True)(atoms).positions
        return (atoms.update_positions(atoms.get_positions() - jnp.asarray(self.learning_rate) * grads),
                grads)

    def minimize(self, atoms: AtomsX, max_steps: int = 1000, tol: float = 1e-2):
        from tqdm import tqdm, trange
        bar = trange(0, max_steps)
        for n in tqdm(range(max_steps)):
            if atoms.neighbors.overflow:
                raise RuntimeError(f'Neighborhood overflow after {n} steps.')
            atoms, grads = self.step(atoms)
            grads_norm = jnp.linalg.norm(grads, axis=-1).max()
            bar.set_description(f"Max. norm forces: {np.array(grads_norm).item():10.5f} (eV/Ang)", refresh=True)
            if grads_norm < tol:
                return atoms, grads

        raise RuntimeError('Optimization did not converge!')


@struct.dataclass
class LBFGS:
    potential: MachineLearningPotential = struct.field(pytree_node=False)
    save_dir: str = struct.field(pytree_node=False)

    step_fn: Callable = struct.field(pytree_node=False)
    optimizer: jaxopt.LBFGS = struct.field(pytree_node=False)

    @classmethod
    def create(cls,
               atoms,
               potential: MachineLearningPotential,
               save_dir: str = None
               ):

        def f_opj(x):
            # TODO: save the whole optimization using id_tab here
            a = atoms.update_positions(x['positions'].reshape(-1, 3))
            return potential(a.to_graph()).sum().astype(x['positions'].dtype), atoms.neighbors.overflow

        opt = jaxopt.LBFGS(f_opj,
                           maxiter=1,
                           implicit_diff=False,
                           unroll=False,
                           has_aux=True,
                           **_get_lbfgs_defaults())

        @jax.jit
        def take_step(p, state):
            step = opt.update(p, state)
            return step.params, step.state

        return LBFGS(potential=potential,
                     save_dir=save_dir,
                     optimizer=opt,
                     step_fn=take_step
                     )

    def step(self, atoms: AtomsX) -> (AtomsX, jnp.ndarray):
        pass

    def minimize(self, atoms: AtomsX, max_steps: int, tol: float = 1e-3):
        from tqdm import trange

        params = {'positions': atoms.get_positions().reshape(-1)}
        opt_state = self.optimizer.init_state(init_params=params)

        bar = trange(0, max_steps)
        converged = False
        forces = None
        max_force_norm = None

        for _ in bar:
            params, opt_state = self.step_fn(params, opt_state)

            if opt_state.aux is True:
                raise RuntimeError('Neighborhood overflow detected! Use a larger capacity multiplier for spatial'
                                   'partitioning on AtomsX.')
                # TODO: Re-start instead of runtime error

            forces = -opt_state.grad['positions'].reshape(-1, 3)
            max_force_norm = jnp.linalg.norm(forces, axis=-1).max()
            bar.set_description(f"Max. norm forces: {np.array(max_force_norm).item():10.5f} (eV/Ang)", refresh=True)
            if max_force_norm < tol:
                converged = True
                break

        atoms_opt = atoms.update_positions(params['positions'].reshape(-1, 3))

        if self.save_dir is not None:
            _save_path = os.path.join(self.save_dir, 'relaxed_structure.h5')
            hdf5_store = HDF5Store(_save_path,
                                   datasets={'positions': DataSetEntry(chunk_length=1,
                                                                       shape=(atoms.get_number_of_atoms(), 3),
                                                                       dtype=np.float32),
                                             'atomic_numbers': DataSetEntry(chunk_length=1,
                                                                            shape=(atoms.get_number_of_atoms(),),
                                                                            dtype=np.int32)
                                             },
                                   mode='w')

            # save the initial structure
            hdf5_store.append({'positions': atoms.get_positions(),
                               'atomic_numbers': atoms.get_atomic_numbers()})

            # save the optimized structure
            hdf5_store.append({'positions': atoms_opt.get_positions(),
                               'atomic_numbers': atoms_opt.get_atomic_numbers()})

            print(f'Saved relaxed structure to {_save_path}.')

        if converged:
            return atoms_opt, forces
        else:
            raise RuntimeError(
                f'BFGS optimization did not converge! {max_force_norm} (max. force norm ) > {tol} (tolerance).')


def _get_lbfgs_defaults():
    return {'linesearch': 'backtracking',
            'condition': 'strong-wolfe',  # options are "armijo", "goldstein", "strong-wolfe" or "wolfe"
            'use_gamma': True,
            'max_stepsize': 0.2
            }
