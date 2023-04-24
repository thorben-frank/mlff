import os
import numpy as np
import dataclasses
import jax
import jax.numpy as jnp
import logging

from jax import lax
from jax.experimental.host_callback import id_tap

from typing import Dict, Tuple
from collections import namedtuple
from tqdm import trange
from ase import units

from .hdfdict import DataSetEntry, HDF5Store


StepData = namedtuple('StepData', ('atoms', 'observables'))


# TODO: save initial atoms object. E.g. use it to initialize IO to h5 and then save directly. Would solve the todo
#  below of determining n_atoms on runtime
@dataclasses.dataclass
class SimulatorX:
    n_atoms: int  # TODO: determine at runtime
    save_dir: str = ''
    save_frequency: int = 1
    save_frequency_dict: Dict[str, int] = None
    temperature_min: float = 1e-1
    temperature_max: float = 1e5
    run_interval: int = 1000

    def __post_init__(self):
        self.trajectory_file = None
        dataset = {
            'positions': DataSetEntry(chunk_length=1, shape=(self.run_interval, self.n_atoms, 3), dtype=np.float32),
            'velocities': DataSetEntry(chunk_length=1, shape=(self.run_interval, self.n_atoms, 3), dtype=np.float32),
            'atomic_numbers': DataSetEntry(chunk_length=1, shape=(self.n_atoms,), dtype=np.int32),
            'temperature': DataSetEntry(chunk_length=1, shape=(self.run_interval,), dtype=np.float32),
            'kinetic_energy': DataSetEntry(chunk_length=1, shape=(self.run_interval,), dtype=np.float32),
            'potential_energy': DataSetEntry(chunk_length=1, shape=(self.run_interval,), dtype=np.float64),
            # 'heat_flux': DataSetEntry(chunk_length=1, shape=(self.run_interval, 3), dtype=np.float64),
            }

        hdf5_store = HDF5Store(os.path.join(self.save_dir, 'trajectory.h5'), datasets=dataset, mode='w')

        def save_trajectory(x: StepData, transform):
            my_dict = {'velocities': x.atoms.get_velocities(),
                       'atomic_numbers': x.atoms.get_atomic_numbers(),
                       'positions': x.atoms.get_positions()
                       }

            my_dict.update(x.observables)
            my_dict_np = jax.tree_map(lambda u: np.asarray(u), my_dict)
            hdf5_store.append(my_dict_np)

        self.save_trajectory = save_trajectory

        def calculate_stats(atoms, properties):
            e_kin = atoms.get_kinetic_energy()
            temperature = atoms.get_temperature() / jnp.asarray(units.kB)
            potential_energy = properties['energy']
            stats = {'kinetic_energy': e_kin,
                     'temperature': temperature,
                     'potential_energy': potential_energy}
            return stats

        def _step(x, aux):
            integrator, atoms = x
            integrator_new, atoms_new, properties = integrator.step(atoms)
            return (integrator_new, atoms_new), StepData(atoms=atoms_new,
                                                         observables=calculate_stats(atoms_new, properties))

        @jax.jit
        def take_steps(x):
            return lax.scan(_step, xs=None, init=x, length=self.run_interval)

        self.step = take_steps

    def run(self, integrator, atoms, steps):

        bar = trange(0, int(steps / self.run_interval))
        delta_t = integrator.timestep / (1000 * units.fs) * self.run_interval

        # wandb.init(project='MD_run')
        overflow_count = 0
        bar.set_description(f"Time: {0:10.3f} (ps)", refresh=True)
        for step in bar:
            final, step_data = self.step((integrator, atoms))
            final_integrator, final_atoms = final

            neighbors_overflow, unfolding_overflow = check_overflow(step_data.atoms)

            # Overflow in the neighborhood list and *no* overflow in unfolding (be it since no unfolding is defined
            # or no spatial unfolding is present)
            if neighbors_overflow and not unfolding_overflow:
                logging.warning('Spatial partitioning overflow detected! Re-allocating neighborhood list.')

                atoms = re_allocate_spatial_partitioning(atoms, step_data)
                overflow_count += 1
                self.clear_cache()

            # Overflow spatial unfolding
            elif unfolding_overflow:
                logging.warning('Spatial unfolding overflow detected! Re-allocating unfolding.')
                atoms = re_allocate_spatial_unfolding(atoms, step_data)
                overflow_count += 1
                self.clear_cache()
            else:
                atoms = final_atoms

            integrator = final_integrator

            if step % self.save_frequency == 0:
                id_tap(self.save_trajectory, step_data)

            t = delta_t * (step + 1 - overflow_count)
            temp = step_data.observables['temperature'][-1].item()
            bar.set_description(f"Time: {t:10.3f} (ps), Temp: {temp:10.3f}", refresh=True)
            if temp > 1e5:
                raise RuntimeError('Temperature > 100_000 K.')
            if temp < 0.001:
                raise RuntimeError('Temperature < 0.001 K.')

    def clear_cache(self):
        self.step._clear_cache()
        assert self.step._cache_size() == 0


def check_overflow(x) -> Tuple:
    neighbors_overflow = x.neighbors.overflow.any()
    # if x.spatial_unfolding is not None:
    #     unfolding_overflow = x.unfolded.overflow.any()
    # else:
    #     unfolding_overflow = None

    return neighbors_overflow, False,  # unfolding_overflow


def re_allocate_spatial_partitioning(atoms, step_data):
    # find one of the overflown atoms and the corresponding positions
    overflown_positions = step_data.atoms.get_positions()[np.nonzero(step_data.atoms.neighbors.overflow)][0]
    # create a new atoms object with the overflown positions
    re_allocate_atoms = atoms.update_positions(overflown_positions, update_neighbors=False)
    # re-allocate the neighborhood list, with the same cutoff and skin but with increased capacity
    spatial_partitioning = atoms.spatial_partitioning
    re_allocate_atoms = re_allocate_atoms.init_spatial_partitioning(cutoff=spatial_partitioning.cutoff,
                                                                    skin=spatial_partitioning.skin,
                                                                    capacity_multiplier=1.25)
    # go back to the positions prior to the MD steps. The resulting atoms object now corresponds to the
    # atoms object before the scanned MD steps with a neighborhood list that has increased
    # capacity_multiplier
    atoms = re_allocate_atoms.update_positions(atoms.positions)
    return atoms
#
#
# def re_allocate_spatial_unfolding(atoms, step_data: StepData):
#     # find one of the overflown atoms and the corresponding positions
#     overflown_positions = step_data.atoms.get_positions()[np.nonzero(step_data.atoms.unfolded.overflow)][0]
#     # create a new atoms object with the overflown positions
#     re_allocate_atoms = atoms.update_positions(overflown_positions, update_neighbors=False)
#     # re-allocate the neighborhood list, with the same cutoff and skin but with increased capacity
#     spatial_partitioning = atoms.spatial_partitioning
#     spatial_unfolding = atoms.spatial_unfolding
#
#     re_allocate_atoms = re_allocate_atoms.reset_spatial_partitioning()
#     re_allocate_atoms = re_allocate_atoms.init_spatial_unfolding(cutoff=spatial_unfolding.cutoff,
#                                                                  skin=spatial_unfolding.skin + 0.1)
#     re_allocate_atoms = re_allocate_atoms.init_spatial_partitioning(cutoff=spatial_partitioning.cutoff,
#                                                                     skin=spatial_partitioning.skin,
#                                                                     capacity_multiplier=1.25)
#     # go back to the positions prior to the MD steps. The resulting atoms object now corresponds to the
#     # atoms object before the scanned MD steps with a neighborhood list that has increased
#     # capacity_multiplier
#     atoms = re_allocate_atoms.update_positions(atoms.positions)
#     return atoms


def re_allocate_spatial_unfolding(atoms, *args, **kwarga):
    return atoms

