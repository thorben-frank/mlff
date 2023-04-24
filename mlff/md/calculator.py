import jax
import jax.numpy as jnp
import numpy as np
import logging
import os

from typing import Any, Dict

from ase.calculators.calculator import Calculator
from ase.neighborlist import primitive_neighbor_list

from functools import partial

from flax.core.frozen_dict import FrozenDict
from flax.training import checkpoints

from mlff.nn.stacknet import get_obs_and_force_fn, get_energy_force_stress_fn, init_stack_net, get_observable_fn
from mlff.geometric import coordinates_to_distance_matrix, coordinates_to_distance_matrix_mic
from mlff.padding.padding import pad_indices
from mlff.io import read_json


logging.basicConfig(level=logging.INFO)

StackNet = Any


class mlffCalculator(Calculator):
    implemented_properties = ['energy', 'forces', 'stress', 'free_energy']

    @classmethod
    def create_from_ckpt_dir(cls,
                             ckpt_dir: str,
                             r_cut: int = None,
                             mic: str = None,
                             calculate_stress: bool = False,
                             E_to_eV: float = 1.,
                             F_to_eV_Ang: float = 1.,
                             dtype: np.dtype = np.float64,
                             *args,
                             **kwargs):

        net = init_stack_net(read_json(os.path.join(ckpt_dir, 'hyperparameters.json')))
        scales = read_json(os.path.join(ckpt_dir, 'scales.json'))
        params = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=None, prefix='checkpoint_loss_')['params']

        return cls(params=params,
                   stack_net=net,
                   scales=scales,
                   r_cut=r_cut,
                   mic=mic,
                   calculate_stress=calculate_stress,
                   E_to_eV=E_to_eV,
                   F_to_eV_Ang=F_to_eV_Ang,
                   dtype=dtype,
                   *args,
                   **kwargs)

    def __init__(
            self,
            params: FrozenDict,
            stack_net: StackNet,
            scales: Dict,
            E_to_eV: float = 1.,
            F_to_eV_Ang: float = 1.,
            n_interactions_max: int = None,
            r_cut: int = None,
            mic: str = None,
            calculate_stress: bool = False,
            dtype: np.dtype = np.float64,
            *args,
            **kwargs
    ):
        """
        ASE calculator given a StackNet and parameters.

        A calculator takes atomic numbers and atomic positions from an Atoms object and calculates the energy and
        forces.

        Args:
            params (FrozenDict): Parameters of the model.
            stack_net (StackNet): An initialized StackNet.
            E_to_eV (float): Conversion factor from whatever energy unit is used by the model to eV.
                By default this parameter is set to convert from kcal/mol.
            F_to_eV_Ang (float): Conversion factor from whatever length unit is used by the model to Angstrom. By
                default, the length unit is not converted (assumed to be in Angstrom)
            *args ():
            **kwargs ():
        """
        super(mlffCalculator, self).__init__(*args, **kwargs)
        self.log = logging.getLogger(__name__)
        self.log.warning(
            'Please remember to specify the proper conversion factors, if your model does not use '
            '\'eV\' and \'Ang\' as units.'
        )
        self.stack_net = stack_net
        self.prop_keys = stack_net.prop_keys
        self.energy_key = self.prop_keys['energy']
        self.force_key = self.prop_keys['force']
        self.R_key = self.prop_keys['atomic_position']
        self.z_key = self.prop_keys['atomic_type']

        self.scales = scales
        self.calculate_stress = calculate_stress
        self.dtype = dtype

        def scale(k, v):
            return np.asarray(self.scales[k]['scale'], self.dtype) * v

        def shift(k, v, z):
            return v + np.array(self.scales[k]['per_atom_shift'], self.dtype)[z].sum()

        def scale_and_shift_fn(x: Dict, z: np.ndarray):
            return {k: shift(k, scale(k, v), z) for (k, v) in x.items()}

        self.scale_and_shift_fn = scale_and_shift_fn

        self.mic = mic
        if self.mic is not None:
            self.unit_cell_key = self.prop_keys.get('unit_cell')
            if self.mic == 'bins':
                self.cell_offset_key = self.prop_keys.get('cell_offset')

        if self.calculate_stress:
            obs_fn = get_energy_force_stress_fn(stack_net)
            self.stress_key = self.prop_keys['stress']
            self.voigt_indices = (
                np.array([0, 1, 2, 1, 0, 0]),
                np.array([0, 1, 2, 2, 2, 1])
            )
        else:
            obs_fn = get_obs_and_force_fn(stack_net)

        model_cut = None
        for g in stack_net.geometry_embeddings:
            if g.module_name == 'geometry_embed':
                model_cut = g.r_cut

        if r_cut is None:
            self.r_cut = model_cut
        else:
            self.r_cut = r_cut

        if model_cut is not None and self.r_cut is not None:
            if model_cut > self.r_cut:
                logging.warning('Model cut is larger than the cut for the neighborhood list. This will '
                                'likely lead to wrong results.')

        if self.r_cut is None and model_cut is None:
            msg = 'Did not find information about the cutoff radius. Setting it to `np.inf` for the index ' \
                  'calculation.'
            logging.warning(msg)

        logging.info('Model cut is {} Ang and neighborhood list cutoff is {} Ang'.format(model_cut, self.r_cut))

        # converts energy from the unit used by the model to eV.
        self.E_to_eV = E_to_eV

        # converts force from the unit used by the model to eV/Ang.
        self.F_to_eV_Ang = F_to_eV_Ang

        # converts lengths to unit used in model.
        self.Ang_to_R = F_to_eV_Ang / E_to_eV

        def energy_and_force_fn(x: Dict):
            x[self.R_key] = x[self.R_key] * self.Ang_to_R
            y = obs_fn(params, x)

            return {'energy': y[self.energy_key],
                    'force': y[self.force_key]}

        self.energy_and_force_fn = jax.jit(jax.vmap(energy_and_force_fn))

        def energy_and_force_and_stress_fn(x: Dict):
            x[self.R_key] = x[self.R_key] * self.Ang_to_R
            y = obs_fn(params, x)

            return {'energy': y[self.energy_key],
                    'force': y[self.force_key],
                    'stress': y[self.stress_key]}

        self.energy_and_force_and_stress_fn = jax.jit(jax.vmap(energy_and_force_and_stress_fn))

        self.idx_pad_fn = lambda x: x
        if n_interactions_max is not None:
            if self.mic == 'bins':
                def _pad_shift(_shift, n_pair_max, pad_value=0):
                    n_pair = _shift.shape[-2]

                    pad_length = n_pair_max - n_pair
                    assert pad_length >= 0

                    pad = partial(np.pad, pad_width=((0, 0), (0, pad_length), (0, 0)), mode='constant',
                                  constant_values=((0, 0), (0, pad_value), (0, 0)))
                    pad_s = pad(_shift)
                    return pad_s

                def _idx_pad_fn(x: Dict) -> Dict:
                    idx_i, idx_j = pad_indices(idx_i=x['idx_i'],
                                               idx_j=x['idx_j'],
                                               n_pair_max=n_interactions_max,
                                               pad_value=-1)
                    cell_offsets = _pad_shift(_shift=x['cell_offset'], n_pair_max=n_interactions_max)
                    return {'idx_i': idx_i, 'idx_j': idx_j, 'cell_offset': cell_offsets}
            else:
                def _idx_pad_fn(x: Dict) -> Dict:
                    idx_i, idx_j = pad_indices(idx_i=x['idx_i'],
                                               idx_j=x['idx_j'],
                                               n_pair_max=n_interactions_max,
                                               pad_value=-1)
                    return {'idx_i': idx_i, 'idx_j': idx_j}
            self.idx_pad_fn = _idx_pad_fn

    def calculate(self, atoms=None, *args, **kwargs):
        super(mlffCalculator, self).calculate(atoms, *args, **kwargs)

        R = np.array(atoms.get_positions())  # shape: (n,3)
        z = np.array(atoms.get_atomic_numbers())  # shape: (n)
        # from mlff.geometric import coordinates_to_distance_matrix
        # logging.warning(coordinates_to_distance_matrix(R[None]))
        cell = None
        if self.mic is not None:
            cell = np.array(atoms.get_cell())  # shape: (3,3)
            pbc = np.array(atoms.get_pbc())  # shape: (3)
            if self.mic == "bins":
                neigh_indx = _md_get_pbc_indices(R=R[None], r_cut=self.r_cut, cell=cell[None], pbc=pbc)
                neigh_indx = jax.tree_map(lambda x: x.squeeze(), neigh_indx)
            elif self.mic == 'naive':
                assert pbc.sum() == 3
                assert self.r_cut < 0.5 * min(np.linalg.norm(cell, axis=-1))
                neigh_indx = _get_md_indices(R=R[None], z=z[None], r_cut=self.r_cut, cell=cell[None], mic=True)
            else:
                raise ValueError(f"{self.mic} is not a valid attribute for `mic` in `mlffCalculator`.")

        else:
            neigh_indx = _get_md_indices(R=R[None], z=z[None], r_cut=self.r_cut)

        neigh_indx = jax.tree_map(lambda x: x[None], neigh_indx)  # add batch dimension
        neigh_indx = self.idx_pad_fn(neigh_indx)  # pad the indices to avoid recompiling

        input_dict = {self.R_key: R, self.z_key: z}
        input_dict = jax.tree_map(lambda x: x[None], input_dict)  # add batch dimension
        input_dict.update(neigh_indx)

        if self.mic is not None:
            input_dict.update({self.unit_cell_key: cell[None]})

        if self.calculate_stress:
            energy_and_forces_and_stress = self.energy_and_force_and_stress_fn(input_dict)
            nn_out = self.scale_and_shift_fn(
                {'energy': np.array(energy_and_forces_and_stress['energy'].reshape(-1), dtype=self.dtype),
                 'force': np.array(energy_and_forces_and_stress['force'].reshape(-1, 3), dtype=self.dtype),
                 'stress': np.array(energy_and_forces_and_stress['stress'].reshape(3, 3), dtype=self.dtype)
                 }, z=z.reshape(-1).astype(np.int16))

            volume = (np.cross(cell[0, :], cell[1, :]) * cell[2, :]).sum()
            self.results = {'energy': nn_out['energy'] * self.E_to_eV,
                            'forces': nn_out['force'] * self.F_to_eV_Ang,
                            'stress': nn_out['stress'][self.voigt_indices] / volume * self.F_to_eV_Ang,
                            'free_energy': nn_out['energy'] * self.E_to_eV}
        else:
            energy_and_forces = self.energy_and_force_fn(input_dict)
            nn_out = self.scale_and_shift_fn(
                {'energy': np.array(energy_and_forces['energy'].reshape(-1), dtype=self.dtype),
                 'force': np.array(energy_and_forces['force'].reshape(-1, 3), dtype=self.dtype)
                 }, z=z.reshape(-1).astype(np.int16))

            self.results = {'energy': nn_out['energy'] * self.E_to_eV,
                            'forces': nn_out['force'] * self.F_to_eV_Ang,
                            'free_energy': nn_out['energy'] * self.E_to_eV}


def _get_md_indices(R: np.ndarray, z: np.ndarray, r_cut: float, cell: np.ndarray = None, mic: bool = False):
    """
        For the `n_data` data frames, return index lists for centering and neighboring atoms given some cutoff radius
        `r_cut` for each structure. As atoms may leave or enter the neighborhood for a given atom within the dataset,
        one can have different lengths for the index lists, even for same structures. Thus, the index lists are padded
        wrt the length `n_pairs_max` of the longest index list observed over all frames in the coordinates `R` across
        all structures. Note, that this is suboptimal if one has a wide range number of atoms in the same dataset. The
        coordinates `R` and the atomic types `z` can also be already padded (see `mlff.padding.padding`) and are
        assumed to be padded with `0` if padded. Index values are padded with `-1`.

        Args:
            R (Array): Atomic coordinates, shape: (n_data,n,3)
            z (Array): Atomic types, shape: (n_data, n)
            r_cut (float): Cutoff distance
            cell (Array): Unit cell with lattice vectors along rows (ASE default), shape: (n_data,3,3)
            mic (bool): Minimal image convention for periodic boundary conditions

        Returns: Tuple of centering and neighboring indices, shape: Tuple[(n_pairs_max), (n_pairs_max)]

        """

    n = R.shape[-2]
    n_data = R.shape[0]
    idx = np.indices((n, n))

    if mic:
        cell_lengths = np.linalg.norm(cell, axis=-1).reshape(-1)  # shape: (n_data*3)
        if r_cut < 0.5 * min(cell_lengths):
            distance_fn = lambda r, c: coordinates_to_distance_matrix_mic(r, c)
        else:
            raise NotImplementedError(f'Minimal image convention currently only implemented for '
                                      f'r_cut < 0.5*min(cell_lengths), but r_cut={r_cut} and 0.5*min(cell_lengths) = '
                                      f'{0.5 * min(cell_lengths)}. Consider using `get_pbc_indices` which uses ASE under '
                                      f'the hood. However, the latter takes ~15 times longer so maybe reduce r_cut.')
    else:
        distance_fn = lambda r, _: coordinates_to_distance_matrix(r)
        cell = np.zeros(len(R))

    @jax.jit
    def neigh_filter(_z, _R, _c):
        Dij = distance_fn(_R, _c).squeeze(axis=-1)  # shape: (n,n)
        return jnp.where((Dij <= r_cut) & (Dij > 0), True, False)

    def get_idx(i):
        idx_ = idx[:, neigh_filter(z[i], R[i], cell[i])]  # shape: (2,n_pairs)
        return idx_

    pad_idx_i, pad_idx_j = map(np.squeeze,
                               np.split(np.array(list(map(get_idx, range(n_data)))),
                                        indices_or_sections=2,
                                        axis=-2))

    return {'idx_i': pad_idx_i, 'idx_j': pad_idx_j}


def _md_get_pbc_indices(R: np.ndarray,
                        cell: np.ndarray,
                        r_cut: float,
                        pbc) -> Dict:
    """
    Get PBC indices as well as corresponding cell shifts.
    Args:
        R (Array): Atomic coordinates, shape: (n,3)
        cell (Array): Unit cell, shape: (3,3). Cell vectors are row-wise.
        r_cut (float): Cutoff radius.
        pbc (Array): Directions in which periodic boundary conditions shall be computed.
    Returns: Dictionary containing the indices for atom i and j as well as the cell shifts. All indices are padded
        to the larges number of indices found in the data.
    """

    quantities = 'ijS'
    idx_i, idx_j, shift = ([], [], [])

    for pos, c in zip(R, cell):
        _idx_i, _idx_j, _s = primitive_neighbor_list(quantities, pbc, c, pos, r_cut, numbers=None,
                                                     self_interaction=False, use_scaled_positions=False,
                                                     max_nbins=1000000.0)
        idx_i += [_idx_i]
        idx_j += [_idx_j]
        shift += [_s]

        def _pad_index_list(idx, n_pair_max, pad_value=-1):
            n_pair = idx.shape[-1]

            pad_length = n_pair_max - n_pair
            assert pad_length >= 0

            pad = partial(np.pad, pad_width=(0, pad_length), mode='constant',
                          constant_values=(0, pad_value))
            pad_idx = pad(idx)
            return pad_idx

        def _pad_shift(_shift, n_pair_max, pad_value=0):
            n_pair = _shift.shape[0]

            pad_length = n_pair_max - n_pair
            assert pad_length >= 0

            pad = partial(np.pad, pad_width=((0, pad_length), (0, 0)), mode='constant',
                          constant_values=((0, pad_value), (0, 0)))
            pad_s = pad(_shift)
            return pad_s

        max_idx_i = max([len(x) for x in idx_i])
        max_idx_j = max([len(x) for x in idx_j])
        max_shift = max([len(x) for x in shift])
        assert max_idx_i == max_idx_j
        assert max_idx_i == max_shift

        idx_pad_fn = partial(_pad_index_list, n_pair_max=max_idx_i)
        shift_pad_fn = partial(_pad_shift, n_pair_max=max_idx_i)

        padded_idx_i = np.stack([idx_pad_fn(x) for x in idx_i], axis=0)
        padded_idx_j = np.stack([idx_pad_fn(x) for x in idx_j], axis=0)
        padded_shift = np.stack([shift_pad_fn(x) for x in shift], axis=0)

        return {'idx_i': padded_idx_i, 'idx_j': padded_idx_j, 'cell_offset': padded_shift}


def make_phonon_calculator(ckpt_dir, n_replicas, **kwargs):
    from glp.vibes import calculator

    vibes_calc = calculator(potential={'potential': 'mlff', 'ckpt_dir': ckpt_dir},
                            calculate={'calculate': 'supercell',
                                       'n_replicas': n_replicas},
                            **kwargs
                            )
    return vibes_calc
