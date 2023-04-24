from ase import units
from ase.md import MDLogger
from ase.md.velocitydistribution import (MaxwellBoltzmannDistribution, Stationary, ZeroRotation)
from ase.io import Trajectory

from pathlib import Path
from tqdm import trange

import numpy as np


class NeuralMDLogger(MDLogger):
    def __init__(self,
                 *args,
                 start_time=0,
                 verbose=False,
                 **kwargs):
        if start_time == 0:
            header = True
        else:
            header = False
        super().__init__(header=header, *args, **kwargs)
        """
        Logger uses ps units.
        """
        self.start_time = start_time
        self.verbose = verbose
        if verbose:
            print(self.hdr)
        self.natoms = self.atoms.get_number_of_atoms()

    def __call__(self):
        if self.start_time > 0 and self.dyn.get_time() == 0:
            return
        epot = self.atoms.get_potential_energy()
        ekin = self.atoms.get_kinetic_energy()
        temp = ekin / (1.5 * units.kB * self.natoms)
        if self.peratom:
            epot /= self.natoms
            ekin /= self.natoms
        if self.dyn is not None:
            t = self.dyn.get_time() / (1000 * units.fs) + self.start_time
            dat = (t,)
        else:
            dat = ()
        dat += (epot + ekin, epot, ekin, temp)
        if self.stress:
            dat += tuple(self.atoms.get_stress() / units.GPa)
        self.logfile.write(self.fmt % dat)
        self.logfile.flush()

        if self.verbose:
            print(self.fmt % dat)


class Simulator:
    def __init__(self,
                 atoms,
                 integrator,
                 T_init,
                 zero_linear_momentum: bool = True,
                 zero_angular_momentum: bool = True,
                 start_time: float = 0,
                 save_dir: str = '.',
                 restart: bool = False,
                 save_frequency: int = 100,
                 min_temp: float = 0.1,
                 max_temp: float = 100000,
                 seed: int = 0):
        self.atoms = atoms
        self.start_time = start_time
        self.integrator = integrator
        self.zero_linear_momentum = zero_linear_momentum
        self.zero_angular_momentum = zero_angular_momentum
        self.save_dir = Path(save_dir)
        self.min_temp = min_temp
        self.max_temp = max_temp
        self.natoms = self.atoms.get_number_of_atoms()
        self.save_frequency = save_frequency

        # intialize system momentum
        if not restart:
            assert (self.atoms.get_momenta() == 0).all()
            MaxwellBoltzmannDistribution(self.atoms, T_init, rng=np.random.default_rng(seed))
            if self.zero_linear_momentum:
                Stationary(self.atoms)  # zero linear momentum
            if self.zero_angular_momentum:
                ZeroRotation(self.atoms)  # zero angular momentum
        # attach trajectory dump
        self.traj = Trajectory(self.save_dir / 'atom.traj', 'a', self.atoms)
        self.integrator.attach(self.traj.write, interval=save_frequency)

        # attach log file
        self.integrator.attach(NeuralMDLogger(self.integrator, self.atoms,
                                              self.save_dir / 'thermo.log',
                                              start_time=start_time, mode='a'),
                               interval=save_frequency)

    def run(self, steps):
        early_stop = False
        step = 0
        bar = trange(steps)

        for step in bar:
            self.integrator.run(1)

            t = self.integrator.get_time() / (1000 * units.fs) + self.start_time
            ekin = self.atoms.get_kinetic_energy()
            temp = ekin / (1.5 * units.kB * self.natoms)

            if step % 100 == 0:
                bar.set_description(f"E_kin: {ekin:10.3f}, Temp: {temp:10.3f}, Time: {t:10.3f} (ps)", refresh=True)

            if temp < self.min_temp or temp > self.max_temp:
                print(f'Temprature {temp:.2f} is out of range: \
                        [{self.min_temp:.2f}, {self.max_temp:.2f}]. \
                        Early stopping the simulation.')
                early_stop = True
                break

        self.traj.close()
        return early_stop, (step + 1)
