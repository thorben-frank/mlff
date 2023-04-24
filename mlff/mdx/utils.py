import jax.numpy as jnp

from .atoms import AtomsX


def zero_rotation(atoms: AtomsX, preserve_temperature: bool = True) -> AtomsX:
    """
    Rescale momentum such that no rotation is present. Inspired by ASE code.

    Args:
        atoms (AtomsX):
        preserve_temperature (bool):

    Returns: Updated AtomsX object with total angular momentum equal to zero.

    """
    T0 = atoms.get_temperature()

    # Find the principal moments of inertia and principal axes basis vectors
    Ip, basis = atoms.get_moments_of_inertia(vectors=True)
    # Calculate the total angular momentum and transform to principal basis
    Lp = jnp.dot(basis, atoms.get_angular_momentum())
    # Calculate the rotation velocity vector in the principal basis, avoiding
    # zero division, and transform it back to the cartesian coordinate system
    omega = jnp.dot(jnp.linalg.inv(basis), jnp.select([Ip > 0], [Lp / Ip]))
    # We subtract a rigid rotation corresponding to this rotation vector
    com = atoms.get_center_of_mass()
    pos = atoms.get_positions()
    pos -= com  # translate center of mass to origin
    velocities = atoms.get_velocities()
    atoms = atoms.update_velocities(velocities - jnp.cross(omega, pos))

    if preserve_temperature:
        atoms = scale_momenta(atoms, T0)

    return atoms


def zero_translation(atoms: AtomsX, preserve_temperature: bool = True):
    """
    Sets the center-of-mass momentum to zero.

    Args:
        atoms (AtomsX):
        preserve_temperature (bool):

    Returns: Updated AtomsX object with removed momenta center of mass.

    """

    # Save initial temperature
    T0 = atoms.get_temperature()

    p = atoms.get_momenta()
    p0 = jnp.sum(p, 0)

    # We should add a constant velocity, not momentum, to the atoms
    masses = atoms.get_masses()
    masses_tot = jnp.sum(masses)
    v0 = p0 / masses_tot
    p -= v0 * masses[:, None]
    atoms = atoms.update_momenta(p)

    if preserve_temperature:
        atoms = scale_momenta(atoms, T0)

    return atoms


def scale_momenta(atoms: AtomsX, T0: float) -> AtomsX:
    """
    Trivial re-scaling of the momenta of the atoms, such that the temperature is equal to T.

    Args:
        atoms (AtomsX):
        T0 (float): Target temperature.

    Returns:

    """
    T = atoms.get_temperature()
    c = jnp.sqrt(T0 / T)
    return atoms.update_momenta(c * atoms.get_momenta())
