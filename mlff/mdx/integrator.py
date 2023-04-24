import jax
import jax.numpy as jnp

from collections import namedtuple

from flax import struct
from .atoms import AtomsX
from .calculator import CalculatorX

from typing import Any, Callable

LangevinCoefficients = namedtuple('LangevinCoefficients', ('c1', 'c2', 'c3', 'c4', 'c5'))


@struct.dataclass
class NoseHooverX:
    timestep: float = struct.field(pytree_node=False)
    temperature: float = struct.field(pytree_node=False)
    ttime: float = struct.field(pytree_node=False)
    zeta: float

    calculator: CalculatorX = struct.field(pytree_node=False)

    target_Ekin: Callable[[AtomsX], float] = struct.field(pytree_node=False, default=None)
    Q: Callable[[AtomsX], float] = struct.field(pytree_node=False, default=None)

    @classmethod
    def create(cls, timestep, temperature, ttime, calculator):
        def target_Ekin(atoms):
            n_atoms = atoms.get_number_of_atoms()
            ekin = 0.5 * (3.0 * n_atoms) * temperature
            return ekin

        def Q(atoms):
            return 3.0 * atoms.get_number_of_atoms() * temperature * (ttime * timestep) ** 2

        zeta = 0.0

        return cls(timestep=timestep,
                   temperature=temperature,
                   ttime=ttime,
                   zeta=zeta,
                   calculator=calculator,
                   target_Ekin=target_Ekin,
                   Q=Q
                   )

    def step(self, atoms):
        masses = atoms.get_masses()[:, None]

        # get current acceleration and velocity:
        accel = self.calculator(atoms)['forces'] / masses
        vel = atoms.get_velocities()

        # make full step in position
        x = atoms.get_positions() + vel * self.timestep + (accel - self.zeta * vel) * (0.5 * self.timestep ** 2)
        atoms_displaced = atoms.update_positions(x, update_neighbors=True)

        # record current velocities
        KE_0 = atoms_displaced.get_kinetic_energy()

        # make half a step in velocity
        vel_half = vel + 0.5 * self.timestep * (accel - self.zeta * vel)
        atoms_displaced = atoms_displaced.update_velocities(vel_half)

        # make a full step in accelerations
        properties = self.calculator(atoms_displaced)
        accel = properties['forces'] / masses

        # make a half step in zeta
        _zeta = self.zeta + 0.5 * self.timestep * (1 / self.Q(atoms)) * (KE_0 - self.target_Ekin(atoms))

        # make another half step in zeta
        _zeta = _zeta + 0.5 * self.timestep * (1 / self.Q(atoms)) * (
                    atoms_displaced.get_kinetic_energy() - self.target_Ekin(atoms))

        # make another half step in velocity
        vel = (atoms_displaced.get_velocities() + 0.5 * self.timestep * accel) / (1 + 0.5 * self.timestep * _zeta)
        atoms_displaced = atoms_displaced.update_velocities(vel)

        return self.replace(zeta=_zeta), atoms_displaced, properties


@struct.dataclass
class LangevinX:
    timestep: float = struct.field(pytree_node=False)
    temperature: float = struct.field(pytree_node=False)
    friction: float = struct.field(pytree_node=False)

    calculator: CalculatorX = struct.field(pytree_node=False)
    coefficients: Callable = struct.field(pytree_node=False)
    prng_key: Any

    fixcm: bool = struct.field(pytree_node=False, default=True)

    @classmethod
    def create(cls, timestep, temperature, friction, calculator, fixcm=True, *args, **kwargs):
        def coefficients(atoms):
            sigma = jnp.sqrt(2 * temperature * friction / atoms.get_masses())
            c1 = timestep / 2. - timestep * timestep * friction / 8.
            c2 = timestep * friction / 2 - timestep * timestep * friction * friction / 8.
            c3 = jnp.sqrt(timestep) * sigma / 2. - timestep ** 1.5 * friction * sigma / 8.
            c5 = timestep ** 1.5 * sigma / (2 * jnp.sqrt(3))
            c4 = friction / 2. * c5
            return LangevinCoefficients(c1=c1, c2=c2, c3=c3[:, None], c4=c4[:, None], c5=c5[:, None])

        prng_key = jax.random.PRNGKey(0)

        return cls(timestep=timestep,
                   temperature=temperature,
                   friction=friction,
                   fixcm=fixcm,
                   calculator=calculator,
                   coefficients=coefficients,
                   prng_key=prng_key)

    def step(self, atoms: AtomsX):
        masses = atoms.get_masses()[:, None]

        forces = self.calculator(atoms)['forces']

        vel = atoms.get_velocities()

        coeff = self.coefficients(atoms)

        k0, k1, k2 = jax.random.split(self.prng_key, num=3)

        xi = jax.random.normal(key=k1, shape=(len(atoms.get_atomic_numbers()), 3), dtype=forces.dtype)
        eta = jax.random.normal(key=k2, shape=(len(atoms.get_atomic_numbers()), 3), dtype=forces.dtype)

        rnd_pos = coeff.c5 * eta  # shape: (n,3)
        rnd_vel = coeff.c3 * xi - coeff.c4 * eta  # shape: (n,3)

        if self.fixcm:
            rnd_pos -= rnd_pos.sum(axis=0, keepdims=True) / atoms.get_number_of_atoms()
            rnd_vel -= (rnd_vel * masses).sum(axis=0, keepdims=True) / (masses * atoms.get_number_of_atoms())

        # Take a half step in velocity
        vel += (coeff.c1 * forces / masses - coeff.c2 * vel + rnd_vel)

        # Full step in positions
        x = atoms.get_positions()
        atoms_displaced = atoms.update_positions(x + self.timestep * vel + rnd_pos)

        # recalculate velocities after RATTLE constraints are applied
        vel = (atoms_displaced.get_positions() - x - rnd_pos) / self.timestep

        properties = self.calculator(atoms_displaced)
        forces = properties['forces']

        # Update the velocities
        vel += (coeff.c1 * forces / masses - coeff.c2 * vel + rnd_vel)

        # Second part of RATTLE taken care of here
        atoms_displaced = atoms_displaced.update_momenta(vel * masses)

        return self.replace(prng_key=k0), atoms_displaced, properties


@struct.dataclass
class BAOABLangevinX:
    """
    Implementation of the Langevin Thermostat following https://aip.scitation.org/doi/pdf/10.1063/1.4916312.
    """

    timestep: float = struct.field(pytree_node=False)
    temperature: float = struct.field(pytree_node=False)
    calculator: CalculatorX = struct.field(pytree_node=False)

    gamma: float = struct.field(pytree_node=False)
    prng_key: Any

    fixcm: bool = struct.field(pytree_node=False)

    @classmethod
    def create(cls, timestep, temperature, calculator, gamma: float = 1e-1, fixcm: bool = True, seed: int = 0):
        rng = jax.random.PRNGKey(seed)
        return BAOABLangevinX(timestep=timestep,
                              temperature=temperature,
                              calculator=calculator,
                              gamma=gamma,
                              fixcm=fixcm,
                              prng_key=rng)

    def step(self, atoms: AtomsX):
        def scale_momentum(x: AtomsX):
            # create momentum distribution
            c1 = jnp.exp(- self.gamma * self.timestep)
            c2 = jnp.sqrt(self.temperature * (1 - c1 ** 2))
            p_dist = Normal(mean=c1 * x.get_momenta(),
                            var=c2 ** 2 * x.get_masses()[:, None])

            # sample from momentum distribution
            rnd0, rnd1 = jax.random.split(self.prng_key, num=2)
            p_rnd = p_dist.sample(rnd0)

            if self.fixcm:
                p_rnd -= p_rnd.sum(axis=0, keepdims=True) / x.get_number_of_atoms()

            # update the atoms momenta
            x = x.update_momenta(momenta=p_rnd)

            # return new atoms object and new rng key
            return x, rnd1

        dt_2 = jnp.asarray(0.5) * self.timestep

        forces = self.calculator(atoms)['forces']
        atoms = atoms.update_momenta(atoms.get_momenta() + dt_2 * forces)
        atoms = atoms.update_positions(atoms.get_positions() + dt_2 * atoms.get_velocities())
        atoms, new_prng_key = scale_momentum(atoms)
        atoms = atoms.update_positions(atoms.get_positions() + dt_2 * atoms.get_velocities())
        properties = self.calculator(atoms)
        forces = properties['forces']
        atoms = atoms.update_momenta(atoms.get_momenta() + dt_2 * forces)

        return self.replace(prng_key=new_prng_key), atoms, properties


@struct.dataclass
class Normal:
    mean: jnp.ndarray
    var: jnp.ndarray

    def sample(self, key):
        mu, std = self.mean, jnp.sqrt(self.var)
        return mu + std * jax.random.normal(key, std.shape, dtype=std.dtype)


@struct.dataclass
class BerendsenX:
    timestep: float = struct.field(pytree_node=False)  # in ASE units
    temperature: float = struct.field(pytree_node=False)  # in energy units
    calculator: CalculatorX = struct.field(pytree_node=False)

    time_constant: float = struct.field(pytree_node=False)  # in ASE units
    fixcm: bool = struct.field(pytree_node=False)

    @classmethod
    def create(cls, timestep, temperature, calculator, time_constant, fixcm: bool = True):
        """
        Create integrator using the Berendsen thermostat for velocity scaling.

        Args:
            timestep (): MD time step, in ASE units.
            temperature (): Target temperature in energy units.
            calculator (): Calculator function.
            time_constant (): Time constant, in ASE units.
            fixcm (): Fix center of mass.

        Returns:

        """

        return BerendsenX(timestep=timestep,
                          temperature=temperature,
                          calculator=calculator,
                          time_constant=time_constant,
                          fixcm=fixcm)

    def step(self, atoms: AtomsX):
        def scale_velocities(x: AtomsX):
            tr = self.timestep / self.time_constant
            current_temperature = x.get_temperature()

            scl_temperature = jnp.sqrt(jnp.asarray(1.0) + tr * (self.temperature / current_temperature - 1.0))

            # Limit the velocity scaling to reasonable values
            # scl_temperature = jnp.where(scl_temperature > 1.1, 1.1, scl_temperature)
            # scl_temperature = jnp.where(scl_temperature < 0.9, 0.9, scl_temperature)

            _p = x.get_momenta()
            _p = scl_temperature * _p
            return x.update_momenta(_p)

        masses = atoms.get_masses()[:, None]

        # re-scale the velocities/momenta
        atoms = scale_velocities(atoms)

        forces = self.calculator(atoms)['forces']

        # first half step in momenta
        p = atoms.get_momenta()
        p += 0.5 * self.timestep * forces

        if self.fixcm:
            # calculate the momentum center of mass and correct for it
            p_sum = p.sum(axis=0) / float(len(p))
            p = p - p_sum

        # full step in positions
        atoms = atoms.update_positions(atoms.get_positions() + self.timestep * p / masses, update_neighbors=True)

        # calculate forces after position update
        properties = self.calculator(atoms)
        forces = properties['forces']

        # second half step in momenta
        atoms = atoms.update_momenta(p + 0.5 * self.timestep * forces)

        return self.replace(), atoms, properties


@struct.dataclass
class VelocityVerletX:
    timestep: float = struct.field(pytree_node=False)
    calculator: CalculatorX = struct.field(pytree_node=False)

    @classmethod
    def create(cls, timestep, calculator):
        return cls(timestep=timestep,
                   calculator=calculator)

    def step(self, atoms):
        masses = atoms.get_masses()[:, None]

        forces = self.calculator(atoms)['forces']

        p = atoms.get_momenta()
        p += jnp.asarray(0.5, forces.dtype) * self.timestep * forces

        pos = atoms.get_positions()

        atoms_displaced = atoms.update_positions(pos + self.timestep * p / masses, update_neighbors=True)

        atoms_displaced = atoms_displaced.update_momenta(p)

        properties = self.calculator(atoms_displaced)
        forces = properties['forces']

        atoms_displaced = atoms_displaced.update_momenta(atoms_displaced.get_momenta() +
                                                         jnp.asarray(0.5, forces.dtype) * self.timestep * forces)

        return self.replace(), atoms_displaced, properties
