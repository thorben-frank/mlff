from ase import units
import jax.numpy as jnp
import jraph
import numpy as np
from typing import Dict, Sequence


def unit_conversion(
        x: Sequence[jraph.GraphsTuple],
        energy_unit: float = units.eV,
        length_unit: float = units.Angstrom
):
    _energy_unit = jnp.asarray(energy_unit)
    _length_unit = jnp.asarray(length_unit)
    for g in x:
        g.globals['energy'] = g.globals.get('energy') * _energy_unit
        g.nodes['forces'] = g.nodes.get('forces') * _energy_unit / _length_unit
        g.nodes['positions'] = g.nodes.get('positions') * _length_unit
        yield g


def subtract_atomic_energy_shifts(x: Sequence[jraph.GraphsTuple], atomic_energy_shifts: Dict):
    # Create a NumPy array filled with zeros.
    result_array = np.zeros(118 + 1)

    # Fill the array using the values from the dictionary.
    for key, value in atomic_energy_shifts.items():
        result_array[key] = value

    # Convert to JAX Array.
    atomic_energy_shifts_arr = jnp.array(result_array)

    for g in x:
        atomic_numbers = g.nodes.get('atomic_numbers')
        g.globals['energy'] = g.globals.get('energy') - jnp.take(atomic_energy_shifts_arr, atomic_numbers).sum()
        yield g


def calculate_energy_mean(x: Sequence[jraph.GraphsTuple]):
    rolling_mean = jnp.asarray(0.)
    for n, g in enumerate(x):
        count = n + 1
        energy = g.globals.get('energy')
        if count == 1:
            rolling_mean = rolling_mean + energy / count
        else:
            rolling_mean = (rolling_mean + energy / (count - 1)) / count * (count - 1)

    return rolling_mean


def calculate_average_number_of_nodes(x: Sequence[jraph.GraphsTuple]):
    rolling_mean = jnp.asarray(0, dtype=jnp.int32)
    for n, g in enumerate(x):
        count = n + 1
        num_nodes = len(g.nodes.get('atomic_numbers'))
        if count == 1:
            rolling_mean = (rolling_mean + num_nodes / count)
        else:
            rolling_mean = ((rolling_mean + num_nodes / (count - 1)) / count * (count - 1))

    return rolling_mean
