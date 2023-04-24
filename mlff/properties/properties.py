from .property_names import *

md17_property_keys = {energy: 'E',
                      force: 'F',
                      atomic_position: 'R',
                      atomic_type: 'z',
                      idx_i: 'idx_i',
                      idx_j: 'idx_j',
                      unit_cell: 'unit_cell',
                      pbc: 'pbc',
                      cell_offset: 'cell_offset',
                      stress: 'stress',
                      node_mask: 'node_mask'}

qm7x_property_keys = {energy: 'ePBE0+MBD',
                      force: 'totFOR',
                      atomic_position: 'atXYZ',
                      atomic_type: 'atNUM',
                      hirshfeld_volume: 'hVOL',
                      hirshfeld_volume_ratio: 'hRAT',
                      partial_charge: 'hCHG',
                      total_dipole_moment: 'vDIP',
                      idx_i: 'idx_i',
                      idx_j: 'idx_j',
                      node_mask: 'node_mask'
                      }
