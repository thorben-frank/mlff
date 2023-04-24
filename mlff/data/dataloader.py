import numpy as np
from ase.io import iread
from ase import Atoms
from dataclasses import dataclass
from typing import Dict
from tqdm import tqdm

from mlff.padding import (pad_forces,
                          pad_atomic_types,
                          pad_coordinates)


@dataclass
class AseDataLoader:
    input_file: str
    output_file: str = None
    load_stress: bool = False
    load_energy_and_forces: bool = True

    def load_all(self) -> Dict:
        def extract_positions(x: Atoms):
            return x.get_positions()

        def extract_numbers(x: Atoms):
            return x.get_atomic_numbers()

        def extract_energy(x: Atoms):
            return x.get_potential_energy()

        def extract_forces(x: Atoms):
            return x.get_forces()

        def extract_stress(x: Atoms):
            return x.get_stress(voigt=False)

        def extract_pbc(x: Atoms):
            return x.get_pbc()

        def extract_unit_cell(x: Atoms):
            return np.array(x.get_cell(complete=False))

        pos = []
        nums = []

        energies = []
        forces = []
        stress = []

        cell = []
        pbc = []

        n_max = max(set(map(lambda x: len(x.get_atomic_numbers()), iread(self.input_file))))

        print(f"Read data from {self.input_file} ...")
        for a in tqdm(iread(self.input_file)):
            pos += [pad_coordinates(extract_positions(a)[None], n_max=n_max).squeeze(axis=0)]
            nums += [pad_atomic_types(extract_numbers(a)[None], n_max=n_max).squeeze(axis=0)]
            cell += [extract_unit_cell(a)]
            pbc += [extract_pbc(a)]

            if self.load_energy_and_forces:
                energies += [extract_energy(a)]
                forces += [pad_forces(extract_forces(a)[None], n_max=n_max).squeeze(axis=0)]

            if self.load_stress:
                stress += [extract_stress(a)]

        loaded_data = {'R': np.stack(pos, axis=0),
                       'z': np.stack(nums, axis=0),
                       'pbc': np.stack(pbc, axis=0),
                       'unit_cell': np.stack(cell, axis=0)
                       }
        if self.load_stress:
            loaded_data.update({'stress': np.stack(stress, axis=0)})
        if self.load_energy_and_forces:
            loaded_data.update({'E': np.stack(energies, axis=0).reshape(-1, 1),
                                'F': np.stack(forces, axis=0)})

        node_mask = np.where(loaded_data['z'] > 0, True, False)
        loaded_data.update({'node_mask': node_mask})

        print("... done!")
        if self.output_file is not None:
            print(f'Write data from {self.input_file} to {self.output_file} ...')
            np.savez(self.output_file, **loaded_data)
            print('... done!')

        return loaded_data
