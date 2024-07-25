from ase import units
import jraph
import numpy as np
from typing import Dict, Sequence


def unit_conversion_graph(
        g,
        energy_unit: float = units.eV,
        length_unit: float = units.Angstrom
):
    _energy_unit = np.asarray(energy_unit)
    _length_unit = np.asarray(length_unit)

    g.globals['energy'] = g.globals.get('energy') * _energy_unit
    g.globals['stress'] = g.globals.get('stress') * _energy_unit / np.power(_length_unit, 3)

    g.nodes['forces'] = g.nodes.get('forces') * _energy_unit / _length_unit
    g.nodes['positions'] = g.nodes.get('positions') * _length_unit

    # if g.globals.get('stress') is not None:
    # if not np.isnan(g.globals['stress']).any() is True:
    #     raise NotImplementedError('Unit conversion for stress not implemented yet.')
    return g


def unit_conversion(
        x: Sequence[jraph.GraphsTuple],
        energy_unit: float = units.eV,
        length_unit: float = units.Angstrom
):
    _energy_unit = np.asarray(energy_unit)
    _length_unit = np.asarray(length_unit)
    for g in x:
        g.globals['energy'] = g.globals.get('energy') * _energy_unit
        g.globals['stress'] = g.globals.get('stress') * _energy_unit / np.power(_length_unit, 3)

        g.nodes['forces'] = g.nodes.get('forces') * _energy_unit / _length_unit
        g.nodes['positions'] = g.nodes.get('positions') * _length_unit

        # if g.globals.get('stress') is not None:
        # if not np.isnan(g.globals['stress']).any() is True:
        #     raise NotImplementedError('Unit conversion for stress not implemented yet.')
        yield g


def subtract_atomic_energy_shifts(x: Sequence[jraph.GraphsTuple], atomic_energy_shifts: Dict):
    # Create a NumPy array filled with zeros.
    result_array = np.zeros(118 + 1)

    # Fill the array using the values from the dictionary.
    for key, value in atomic_energy_shifts.items():
        result_array[key] = value

    # Convert to numpy array.
    atomic_energy_shifts_arr = np.array(result_array)

    for g in x:
        atomic_numbers = g.nodes.get('atomic_numbers')
        g.globals['energy'] = g.globals.get('energy') - np.take(atomic_energy_shifts_arr, atomic_numbers).sum()
        yield g


def calculate_energy_mean(x: Sequence[jraph.GraphsTuple]):
    rolling_mean = np.asarray(0.)
    for n, g in enumerate(x):
        count = n + 1
        energy = g.globals.get('energy')
        if count == 1:
            rolling_mean = rolling_mean + energy / count
        else:
            rolling_mean = (rolling_mean + energy / (count - 1)) / count * (count - 1)

    return rolling_mean


def calculate_average_number_of_nodes(x: Sequence[jraph.GraphsTuple]):
    rolling_mean = np.asarray(0.)
    for n, g in enumerate(x):
        count = n + 1
        num_nodes = len(g.nodes.get('atomic_numbers'))
        if count == 1:
            rolling_mean = (rolling_mean + num_nodes / count)
        else:
            rolling_mean = ((rolling_mean + num_nodes / (count - 1)) / count * (count - 1))

    return rolling_mean


def calculate_average_number_of_neighbors(x: Sequence[jraph.GraphsTuple]):
    rolling_mean = np.asarray(0.)
    for n, g in enumerate(x):
        count = n + 1
        num_edges = float(len(g.receivers))
        num_nodes = float(len(g.nodes.get('atomic_numbers')))
        if count == 1:
            rolling_mean = (rolling_mean + (num_edges / num_nodes) / count)
        else:
            rolling_mean = ((rolling_mean + (num_edges / num_nodes) / (count - 1)) / count * (count - 1))

    return rolling_mean
