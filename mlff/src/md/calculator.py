import jax
import logging
import numpy as np

from typing import Any, Dict
from ase.calculators.calculator import Calculator
from ase.units import kcal, mol
from flax.core.frozen_dict import FrozenDict

from mlff.src.nn.stacknet import get_obs_and_force_fn
from mlff.src.indexing.indices import index_padding_length
from mlff.src.geometric import coordinates_to_distance_matrix
from mlff.src.padding.padding import pad_indices

logging.basicConfig(level=logging.INFO)

StackNet = Any


class mlffCalculator(Calculator):
    implemented_properties = ['energy', 'forces']

    def __init__(
            self,
            params: FrozenDict,
            stack_net: StackNet,
            E_to_eV: float = kcal/mol,
            F_to_eV_Ang: float = kcal/mol,
            n_interactions_max: int = None,
            r_cut: int = None,
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

        obs_fn = get_obs_and_force_fn(stack_net)

        model_cut = None
        for g in stack_net.geometry_embeddings:
            if g.module_name == 'geometry_embed':
                model_cut = g.r_cut

        if r_cut is None:
            self.r_cut = model_cut
        else:
            self.r_cut = r_cut
        if r_cut is not None and model_cut is not None:
            if model_cut > r_cut:
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
            return {'energy': y[self.energy_key]*self.E_to_eV,
                    'forces': y[self.force_key]*self.F_to_eV_Ang}

        self.energy_and_force_fn = jax.jit(jax.vmap(energy_and_force_fn))

        self.idx_pad_fn = lambda x: x
        if n_interactions_max is not None:
            def _idx_pad_fn(x: Dict) -> Dict:
                idx_i, idx_j = pad_indices(idx_i=x['idx_i'],
                                           idx_j=x['idx_j'],
                                           n_pair_max=n_interactions_max,
                                           pad_value=-1)
                return {'idx_i': idx_i, 'idx_j': idx_j}
            self.idx_pad_fn = _idx_pad_fn

    def calculate(self, atoms=None, *args, **kwargs):
        super(mlffCalculator, self).calculate(atoms, *args, **kwargs)

        # convert model units to ASE default units
        R = np.array(atoms.get_positions())
        z = np.array(atoms.get_atomic_numbers())
        neigh_indx = _get_md_indices(R=R[None], z=z[None], r_cut=self.r_cut)
        neigh_indx = jax.tree_map(lambda x: x[None], neigh_indx)  # add batch dimension
        neigh_indx = self.idx_pad_fn(neigh_indx)  # pad the indices to avoid recompiling

        input_dict = {self.R_key: R, self.z_key: z}
        input_dict = jax.tree_map(lambda x: x[None], input_dict)  # add batch dimension
        input_dict.update(neigh_indx)

        energy_and_forces = self.energy_and_force_fn(input_dict)

        self.results = {'energy': energy_and_forces['energy'].reshape(-1),
                        'forces': energy_and_forces['forces'].reshape(-1, 3)
                        }


def _get_md_indices(R: np.ndarray, z: np.ndarray, r_cut: float):
    """
        For the `n_data` data frames, return index lists for centering and neighboring atoms given some cutoff radius
        `r_cut` for each structure. As atoms may leave or enter the neighborhood for a given atom within the dataset,
        one can have different lengths for the index lists, even for same structures. Thus, the index lists are padded
        wrt the length `n_pairs_max` of the longest index list observed over all frames in the coordinates `R` across
        all structures. Note, that this is suboptimal if one has a wide range number of atoms in the same dataset. The
        coordinates `R` and the atomic types `z` can also be already padded (see `mlff.src.padding.padding`) and are
        assumed to be padded with `0` if padded. Index values are padded with `-1`.

        Args:
            R (Array): Atomic coordinates, shape: (n_data,n,3)
            z (Array): Atomic types, shape: (n_data, n)
            r_cut (float): Cutoff distance

        Returns: Tuple of centering and neighboring indices, shape: Tuple[(n_pairs_max), (n_pairs_max)]

        """

    n = R.shape[-2]
    n_data = R.shape[0]
    pad_length = index_padding_length(R, z, r_cut)
    idx = np.indices((n, n))

    def get_idx(i):
        Dij = coordinates_to_distance_matrix(R[i]).squeeze(axis=-1)  # shape: (n,n)
        msk_ij = (np.einsum('i, j -> ij', z[i], z[i]) != 0).astype(np.int16)  # shape: (n,n)
        Dij_x_msk_ij = Dij * msk_ij  # shape: (n,n)
        idx_ = idx[:, np.where((Dij_x_msk_ij <= r_cut) & (Dij_x_msk_ij > 0), True, False)]  # shape: (2,n_pairs)
        pad_idx = np.pad(idx_, ((0, 0), (0, int(pad_length[i]))), mode='constant', constant_values=((0, 0), (0, -1)))
        # shape: (2,n_pair+pad_length)
        return pad_idx

    pad_idx_i, pad_idx_j = map(np.squeeze,
                               np.split(np.array(list(map(get_idx, range(n_data)))),
                                        indices_or_sections=2,
                                        axis=-2))

    return {'idx_i': pad_idx_i, 'idx_j': pad_idx_j}