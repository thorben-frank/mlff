import jax
import jax.numpy as jnp
import flax.linen as nn
from jax.ops import segment_sum
from typing import Any, Callable, Dict
from mlff.nn.base.sub_module import BaseSubModule
from typing import Optional
from mlff.cutoff_function import add_cell_offsets_sparse
from mlff.masking.mask import safe_norm
from ase.units import Bohr, Hartree
from ase.units import alpha as fine_structure
from mlff.nn.observable.dispersion_ref_data import alphas, C6_coef
from jax.scipy.special import factorial
from mlff.masking.mask import safe_scale

@jax.jit
def _switch_component(x: jnp.ndarray, ones: jnp.ndarray, zeros: jnp.ndarray) -> jnp.ndarray:
    """ Component of the switch function, only for internal use. """
    x_ = jnp.where(x <= 0, ones, x)  # prevent nan in backprop
    return jnp.where(x <= 0, zeros, jnp.exp(-ones / x_))

@jax.jit
def switch_function(x: jnp.ndarray, cuton: float, cutoff: float) -> jnp.ndarray:
    """
    Switch function that smoothly (and symmetrically) goes from f(x) = 1 to
    f(x) = 0 in the interval from x = cuton to x = cutoff. For x <= cuton,
    f(x) = 1 and for x >= cutoff, f(x) = 0. This switch function has infinitely
    many smooth derivatives.
    NOTE: The implementation with the "_switch_component" function is
    numerically more stable than a simplified version, it is not recommended 
    to change this!
    """
    x = (x - cuton) / (cutoff - cuton)
    ones = jnp.ones_like(x)
    zeros = jnp.zeros_like(x)
    fp = _switch_component(x, ones, zeros)
    fm = _switch_component(1 - x, ones, zeros)
    return jnp.where(x <= 0, ones, jnp.where(x >= 1, zeros, fm / (fp + fm)))

Array = Any

class EnergySparse(BaseSubModule):
    prop_keys: Dict
    zmax: int = 118
    regression_dim: int = None
    activation_fn: Callable[[Any], Any] = lambda u: u
    learn_atomic_type_scales: bool = False
    learn_atomic_type_shifts: bool = False
    zbl_repulsion: bool = False
    zbl_repulsion_shift: float = 0.
    output_is_zero_at_init: bool = True
    output_convention: str = 'per_structure'
    module_name: str = 'energy'
    electrostatic_energy_bool: bool = False
    electrostatic_energy: Optional[Any] = None
    dispersion_energy_bool: bool = False
    dispersion_energy: Optional[Any] = None

    def setup(self):
        if self.output_is_zero_at_init:
            self.kernel_init = nn.initializers.zeros_init()
        else:
            self.kernel_init = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, inputs: Dict, *args, **kwargs):
        """
        #TODO: Update docstring
        Args:
            inputs (Dict):
                x (Array): Node features, (num_nodes, num_features)
                atomic_numbers (Array): Atomic numbers, (num_nodes)
                batch_segments (Array): Batch segments, (num_nodes)
                node_mask (Array): Node mask, (num_nodes)
                graph_mask (Array): Graph mask, (num_graphs)
            *args ():
            **kwargs ():

        Returns:

        """
        x = inputs['x']  # (num_nodes, num_features)
        atomic_numbers = inputs['atomic_numbers']  # (num_nodes)
        batch_segments = inputs['batch_segments']  # (num_nodes)
        node_mask = inputs['node_mask']  # (num_nodes)
        graph_mask = inputs['graph_mask']  # (num_graphs)

        num_graphs = len(graph_mask)
        if self.learn_atomic_type_shifts:
            energy_offset = self.param(
                'energy_offset',
                nn.initializers.zeros_init(),
                (self.zmax + 1, )
            )[atomic_numbers]  # (num_nodes)
        else:
            energy_offset = jnp.zeros((1,), dtype=x.dtype)

        if self.learn_atomic_type_scales:
            atomic_scales = self.param(
                'atomic_scales',
                nn.initializers.ones_init(),
                (self.zmax + 1,)
            )[atomic_numbers]  # (num_nodes)
        else:
            atomic_scales = jnp.ones((1, ), dtype=x.dtype)

        if self.regression_dim is not None:
            y = nn.Dense(
                self.regression_dim,
                kernel_init=nn.initializers.lecun_normal(),
                name='energy_dense_regression'
            )(x)  # (num_nodes, regression_dim)
            y = self.activation_fn(y)  # (num_nodes, regression_dim)
            atomic_energy = nn.Dense(
                1,
                kernel_init=self.kernel_init,
                name='energy_dense_final'
            )(y).squeeze(axis=-1)  # (num_nodes)
        else:
            atomic_energy = nn.Dense(
                1,
                kernel_init=self.kernel_init,
                name='energy_dense_final'
            )(x).squeeze(axis=-1)  # (num_nodes)

        atomic_energy = atomic_energy * atomic_scales
        atomic_energy += energy_offset  # (num_nodes)

        atomic_energy = safe_scale(atomic_energy, node_mask)

        if self.zbl_repulsion:
            raise NotImplementedError('ZBL Repulsion for sparse model not implemented yet.')
        
        if self.electrostatic_energy_bool:
            electrostatic_energy = self.electrostatic_energy(inputs)['electrostatic_energy']
            atomic_energy += electrostatic_energy

        if self.dispersion_energy_bool:
            dispersion_energy = self.dispersion_energy(inputs)['dispersion_energy']
            atomic_energy += dispersion_energy
        if self.output_convention == 'per_structure':
            energy = segment_sum(
                atomic_energy,
                segment_ids=batch_segments,
                num_segments=num_graphs
            )  # (num_graphs)
            energy = safe_scale(energy, graph_mask)

            return dict(energy=energy)
        elif self.output_convention == 'per_atom':
            energy = atomic_energy  # (num_nodes)

            return dict(energy=energy)
        else:
            raise ValueError(
                f'{self.output_convention} is invalid argument for attribute `output_convention`.'
            )

    def reset_output_convention(self, output_convention):
        self.output_convention = output_convention

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {'zmax': self.zmax,
                                   'output_is_zero_at_init': self.output_is_zero_at_init,
                                   'output_convention': self.output_convention,
                                   'zbl_repulsion': self.zbl_repulsion,
                                   'zbl_repulsion_shift': self.zbl_repulsion_shift,
                                   'electrostatic_energy_bool': self.electrostatic_energy_bool,
                                   'dispersion_energy_bool': self.dispersion_energy_bool,
                                   'prop_keys': self.prop_keys}
                }

class DipoleSparse(BaseSubModule):
    prop_keys: Dict
    regression_dim: int = None
    activation_fn: Callable[[Any], Any] = lambda u: u
    output_is_zero_at_init: bool = True
    module_name: str = 'dipole'

    def setup(self):
        # self.partial_charge_key = self.prop_keys.get('partial_charge')
        # self.atomic_type_key = self.prop_keys.get('atomic_type')
        # self.total_charge_key = self.prop_keys.get('total_charge')

        if self.output_is_zero_at_init:
            self.kernel_init = nn.initializers.zeros_init()
        else:
            self.kernel_init = nn.initializers.lecun_normal()        

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs) -> Dict[str, jnp.ndarray]:
        """
        #TODO: Update docstring
        Predict partial charges, from atom-wise features `x`, atomic types `atomic_numbers` and the total charge of the system `total_charge`.

        Args:
            inputs (Dict):
                x (Array): Node features, (num_nodes, num_features)
                atomic_numbers (Array): Atomic numbers, (num_nodes)
                total_charge (Array): Total charge, shape: (1)
                batch_segments (Array): Batch segments, (num_nodes)
                node_mask (Array): Node mask, (num_nodes)
                graph_mask (Array): Graph mask, (num_graphs)
            *args ():
            **kwargs ():

        #Returns: Dictionary of form {'q': Array}, where Array are the predicted partial charges, shape: (n,1)

        """
        x = inputs['x']  # (num_nodes, num_features)
        atomic_numbers = inputs['atomic_numbers']  # (num_nodes)
        batch_segments = inputs['batch_segments']  # (num_nodes)
        node_mask = inputs['node_mask']  # (num_nodes)
        graph_mask = inputs['graph_mask']  # (num_graphs)
        positions = inputs['positions'] # (num_nodes, 3)
        total_charge = inputs['total_charge'] # (num_graphs) #TODO: read total_charge from loaded graph

        num_graphs = len(graph_mask)
        num_nodes = len(node_mask)

        #q_ - element-dependent bias
        q_ = nn.Embed(num_embeddings=100, features=1)(atomic_numbers).squeeze(axis=-1)  # shape: (num_nodes)
        if self.regression_dim is not None:
            y = nn.Dense(
                self.regression_dim,
                kernel_init=nn.initializers.lecun_normal(),
                name='charge_dense_regression'
            )(x)  # (num_nodes, regression_dim)
            y = self.activation_fn(y)  # (num_nodes, regression_dim)
            x_ = nn.Dense(
                1,
                kernel_init=self.kernel_init,
                name='charge_dense_final'
            )(y).squeeze(axis=-1)  # (num_nodes)
        else:
            x_ = nn.Dense(
                1,
                kernel_init=self.kernel_init,
                name='charge_dense_final'
            )(x).squeeze(axis=-1)  # (num_nodes)

        #mask x_q from padding graph
        x_q = safe_scale(x_ + q_, node_mask)

        total_charge_predicted = segment_sum(
            x_q,
            segment_ids=batch_segments,
            num_segments=num_graphs
        )  # (num_graphs)

        _, number_of_atoms_in_molecule = jnp.unique(batch_segments, return_counts = True, size=num_graphs)

        charge_conservation = (1 / number_of_atoms_in_molecule) * (total_charge - total_charge_predicted)
        partial_charges = x_q + jnp.repeat(charge_conservation, number_of_atoms_in_molecule, total_repeat_length = num_nodes)   # shape: (num_nodes)

        mu_i = positions * partial_charges[:, None] #(512,3) * (512,)

        dipole = segment_sum(
            mu_i,
            segment_ids=batch_segments,
            num_segments=num_graphs
        )  # (num_graphs, 3)
        dipole = jnp.linalg.norm(dipole, axis = 1)  # (num_graphs)
        dipole = safe_scale(dipole, graph_mask)

        return dict(dipole=dipole)
    
    def reset_output_convention(self, output_convention):
        self.output_convention = output_convention

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {'output_is_zero_at_init': self.output_is_zero_at_init,
                                   'prop_keys': self.prop_keys}        
                }    
    
class HirshfeldSparse(BaseSubModule):
    prop_keys: Dict
    regression_dim: int = None
    activation_fn: Callable[[Any], Any] = lambda u: u
    output_is_zero_at_init: bool = True
    module_name = 'hirshfeld_ratios'

    def setup(self):
        # self.atomic_type_key = self.prop_keys.get(pn.atomic_type)
        # self.hirshfeld_volume_ratio_key = self.prop_keys.get(pn.hirshfeld_volume_ratio)

        if self.output_is_zero_at_init:
            self.kernel_init = nn.initializers.zeros_init()
        else:
            self.kernel_init = nn.initializers.lecun_normal() 

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs) -> Dict[str, jnp.ndarray]:
        """
        #TODO: Update docstring
        Predict Hirshfeld volumes from atom-wise features `x` and atomic types `z`.

        Args:
            inputs (Dict):
                x (Array): Atomic features, shape: (n,F)
                z (Array): Atomic types, shape: (n)
            *args ():
            **kwargs ():

        Returns: Dictionary of form {'v_eff': Array}, where Array are the predicted Hirshfeld ratios

        """
        x = inputs['x']  # (num_nodes, num_features)
        node_mask = inputs['node_mask']  # (num_nodes)
        atomic_numbers = inputs['atomic_numbers']  # (num_nodes)

        F = x.shape[-1]

        v_shift = nn.Embed(num_embeddings=100, features=1)(atomic_numbers).squeeze(axis=-1)  # shape: (num_nodes)
        q = nn.Embed(num_embeddings=100, features=int(F/2))(atomic_numbers)  # shape: (n,F/2)
        
        if self.regression_dim is not None:
            y = nn.Dense(
                self.regression_dim,
                kernel_init=nn.initializers.lecun_normal(),
                name='hirshfeld_ratios_dense_regression'
            )(x)  # (num_nodes, regression_dim)
            y = self.activation_fn(y)  # (num_nodes, regression_dim)
            k = nn.Dense(
                int(F/2),
                kernel_init=self.kernel_init,
                name='hirshfeld_ratios_dense_final'
            )(y) # (num_nodes)
        else:
            k = nn.Dense(
                int(F/2),
                kernel_init=self.kernel_init,
                name='hirshfeld_ratios_dense_final'
            )(x) # (num_nodes)

        qk = (q * k / jnp.sqrt(k.shape[-1])).sum(axis=-1) 

        v_eff = v_shift + qk  # shape: (n)
        hirshfeld_ratios = safe_scale(jnp.abs(v_eff), node_mask)
        #TODO: better way to ensure positive values?
        #TODO: Check whether hirshfeld ratios make sense
        #TODO: remove abs() in further calls

        return dict(hirshfeld_ratios=hirshfeld_ratios)

    def reset_output_convention(self, output_convention):
        self.output_convention = output_convention
    
class PartialChargesSparse(BaseSubModule):
    prop_keys: Dict
    regression_dim: int = None
    activation_fn: Callable[[Any], Any] = lambda u: u
    output_is_zero_at_init: bool = True
    module_name: str = 'partial_charges'
     
    def setup(self):
        # self.partial_charge_key = self.prop_keys.get('partial_charge')
        # self.atomic_type_key = self.prop_keys.get('atomic_type')
        # self.total_charge_key = self.prop_keys.get('total_charge')
        # self._partial_charges = None

        if self.output_is_zero_at_init:
            self.kernel_init = nn.initializers.zeros_init()
        else:
            self.kernel_init = nn.initializers.lecun_normal()   

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs) -> Dict[str, jnp.ndarray]:

        x = inputs['x']  # (num_nodes, num_features)
        atomic_numbers = inputs['atomic_numbers']  # (num_nodes)
        batch_segments = inputs['batch_segments']  # (num_nodes)
        node_mask = inputs['node_mask']  # (num_nodes)
        graph_mask = inputs['graph_mask']  # (num_graphs)
        total_charge = inputs['total_charge'] # (num_graphs)

        num_graphs = len(graph_mask)
        num_nodes = len(node_mask)

        #q_ - element-dependent bias
        q_ = nn.Embed(num_embeddings=100, features=1)(atomic_numbers).squeeze(axis=-1)  # shape: (num_nodes)
        if self.regression_dim is not None:
            y = nn.Dense(
                self.regression_dim,
                kernel_init=nn.initializers.lecun_normal(),
                name='charge_dense_regression_vec'
            )(x)  # (num_nodes, regression_dim)
            y = self.activation_fn(y)  # (num_nodes, regression_dim)
            x_ = nn.Dense(
                1,
                kernel_init=self.kernel_init,
                name='charge_dense_final_vec'
            )(y).squeeze(axis=-1)  # (num_nodes)
        else:
            x_ = nn.Dense(
                1,
                kernel_init=self.kernel_init,
                name='charge_dense_final_vec'
            )(x).squeeze(axis=-1)  # (num_nodes)

        #mask x_q from padding graph
        x_q = safe_scale(x_ + q_, node_mask)

        total_charge_predicted = segment_sum(
            x_q,
            segment_ids=batch_segments,
            num_segments=num_graphs
        )  # (num_graphs)

        _, number_of_atoms_in_molecule = jnp.unique(batch_segments, return_counts = True, size=num_graphs)

        charge_conservation = (1 / number_of_atoms_in_molecule) * (total_charge - total_charge_predicted)
        partial_charges = x_q + jnp.repeat(charge_conservation, number_of_atoms_in_molecule, total_repeat_length = num_nodes)   # shape: (num_nodes)
        
        #TODO: Check whether partial charges make sense
        #TODO: Constrain or normalize partial charges to not exceed total charge, maybe not needed. Supporting info in SpookyNet shows that partial charges are already reasonable.
        return dict(partial_charges=partial_charges)

    def reset_output_convention(self, output_convention):
        self.output_convention = output_convention
    

class DipoleVecSparse(BaseSubModule):
    prop_keys: Dict
    regression_dim: int = None
    activation_fn: Callable[[Any], Any] = lambda u: u
    output_is_zero_at_init: bool = True
    module_name: str = 'dipole_vec'
    partial_charges: Optional[Any] = None
     
    def setup(self):
        # self.partial_charge_key = self.prop_keys.get('partial_charge')
        # self.atomic_type_key = self.prop_keys.get('atomic_type')
        # self.total_charge_key = self.prop_keys.get('total_charge')
        # self._partial_charges = None

        if self.output_is_zero_at_init:
            self.kernel_init = nn.initializers.zeros_init()
        else:
            self.kernel_init = nn.initializers.lecun_normal()   

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs) -> Dict[str, jnp.ndarray]:

        batch_segments = inputs['batch_segments']  # (num_nodes)
        graph_mask = inputs['graph_mask']  # (num_graphs)
        graph_mask_expanded = inputs['graph_mask_expanded']
        positions = inputs['positions'] # (num_nodes, 3)
        partial_charges = self.partial_charges(inputs)['partial_charges']
        num_graphs = len(graph_mask)

        #TODO: Check whether partial charges make sense
        #TODO: Constrain or normalize partial charges to not exceed total charge, maybe not needed. Supporting info in SpookyNet shows that partial charges are already reasonable.

        mu_i = positions * partial_charges[:, None] #(512,3) * (512,)

        dipole = segment_sum(
            mu_i,
            segment_ids=batch_segments,
            num_segments=num_graphs
        )  # (num_graphs, 3)

        dipole_vec = safe_scale(dipole, graph_mask_expanded)

        return dict(dipole_vec=dipole_vec)

    def reset_output_convention(self, output_convention):
        self.output_convention = output_convention
    
@jax.jit
def _coulomb(q: jnp.ndarray, rij: jnp.ndarray, 
             idx_i: jnp.ndarray, idx_j: jnp.ndarray,
             kehalf: float, cuton: float, cutoff: float
) -> jnp.ndarray:
    """ Pairwise Coulomb interaction """
    fac = kehalf * q[idx_i] * q[idx_j]
    f = switch_function(rij, cuton, cutoff)
    coulomb = 1.0 / rij
    damped = 1.0 / (rij ** 2 + 1.0) ** (1.0 / 2.0)
    pairwise = fac * (f * damped + (1 - f) * coulomb)

    return pairwise

@jax.jit
def _coulomb_erf(q: jnp.ndarray, rij: jnp.ndarray, 
             idx_i: jnp.ndarray, idx_j: jnp.ndarray,
             kehalf: float, sigma: float#, sigma: jnp.ndarray
) -> jnp.ndarray:
    """ Pairwise Coulomb interaction with erf damping """
    pairwise = kehalf * q[idx_i] * q[idx_j] * jax.scipy.special.erf(rij/sigma)/rij
    return pairwise

@jax.jit
def sigma_cubic_fit(alpha):
    vdW_radius = fine_structure**(-4/21)*alpha**(1/7) 
    b0 = -0.00433008
    b1 = 0.24428889
    b2 = 0.04125273
    b3 = -0.00078893
    sigma = b3*vdW_radius**3 + b2*vdW_radius**2 + b1*vdW_radius + b0
    return sigma*jnp.sqrt(2)

class ElectrostaticEnergySparse(BaseSubModule):
    prop_keys: Dict
    regression_dim: int = None
    hirshfeld_ratios: Optional[Any] = None
    activation_fn: Callable[[Any], Any] = lambda u: u
    output_is_zero_at_init: bool = True
    module_name: str = 'electrostatic_energy'
    partial_charges: Optional[Any] = None
    ke: float = 14.399645351950548 #TODO: check if this is the correct value
    cuton: float = 0.25 * 5.
    cutoff: float = 0.75 * 5.
    set_lr_cutoff: Optional[Callable] = None

    #comment: for now cuton and cutoff are set by hand. 
    #lr_cutoff is set to None
    #use_ewald_summation is set to False
    #how to make partial charges not be adjusted by the ElectrostaticEnergySparse/EnergySparse?

    def setup(self):
        self.kehalf = self.ke / 2
        self.lr_cutoff = None
        # self.set_lr_cutoff(self.lr_cutoff)

        # """ Change the long range cutoff. """
        # self.lr_cutoff = lr_cutoff
        # if self.lr_cutoff is not None:
        #     self.lr_cutoff2 = lr_cutoff ** 2
        #     self.two_div_cut = 2.0 / lr_cutoff
        #     self.rcutconstant = lr_cutoff / (lr_cutoff ** 2 + 1.0) ** (3.0 / 2.0)
        #     self.cutconstant = (2 * lr_cutoff ** 2 + 1.0) / (lr_cutoff ** 2 + 1.0) ** (
        #         3.0 / 2.0
        #     )
        # else:
        #     self.lr_cutoff2 = None
        #     self.two_div_cut = None
        #     self.rcutconstant = None
        #     self.cutconstant = None

        # should be turned on manually if the user knows what they are doing
        self.use_ewald_summation = False
        self.alpha = 0.0
        self.alpha2 = 0.0
        self.two_pi = 2.0 * jnp.pi
        self.one_over_sqrtpi = 1 / jnp.sqrt(jnp.pi)
        self.kmul = jnp.array([], dtype=jnp.int32)

    # def reset_parameters(self) -> None:
    #     """ For compatibility with other modules. """
    #     pass

    # def set_kmax(self, Nxmax: int, Nymax: int, Nzmax: int) -> None:
    #     """ Set integer reciprocal space cutoff for Ewald summation """
    #     kx = jnp.arange(0, Nxmax + 1)
    #     kx = jnp.concatenate([kx, -kx[1:]])
    #     ky = jnp.arange(0, Nymax + 1)
    #     ky = jnp.concatenate([ky, -ky[1:]])
    #     kz = jnp.arange(0, Nzmax + 1)
    #     kz = jnp.concatenate([kz, -kz[1:]])
    #     kmul = jnp.array(jnp.cartesian_product(kx, ky, kz)[1:])  # 0th entry is 0 0 0
    #     kmax = max(max(Nxmax, Nymax), Nzmax)
    #     self.kmul = kmul[jnp.sum(kmul ** 2, axis=-1) <= kmax ** 2]

    # def set_alpha(self, alpha: Optional[float] = None) -> None:
    #     """ Set real space damping parameter for Ewald summation """
    #     if alpha is None:  # automatically determine alpha
    #         alpha = 4.0 / self.cutoff + 1e-3
    #     self.alpha = alpha
    #     self.alpha2 = alpha ** 2
    #     self.two_pi = 2.0 * jnp.pi
    #     self.one_over_sqrtpi = 1 / jnp.sqrt(jnp.pi)
    #     # print a warning if alpha is so small that the reciprocal space sum
    #     # might "leak" into the damped part of the real space coulomb interaction
    #     if alpha * self.cutoff < 4.0:  # erfc(4.0) ~ 1e-8
    #         print(
    #             "Warning: Damping parameter alpha is",
    #             alpha,
    #             "but probably should be at least",
    #             4.0 / self.cutoff,
    #         )

    # def _real_space(
    #     self,
    #     N: int,
    #     q: jnp.ndarray,
    #     rij: jnp.ndarray,
    #     idx_i: jnp.ndarray,
    #     idx_j: jnp.ndarray,
    # ) -> jnp.ndarray:
    #     fac = self.kehalf * q[idx_i] * q[idx_j]
    #     f = switch_function(rij, self.cuton, self.cutoff)
    #     coulomb = 1.0 / rij
    #     damped = 1.0 / (rij ** 2 + 1.0) ** (1.0 / 2.0)
    #     pairwise = fac * (f * damped + (1 - f) * coulomb) * jnp.erfc(self.alpha * rij)
    #     return jnp.zeros(N).at[idx_i].add(pairwise)

    # def _reciprocal_space(
    #     self,
    #     q: jnp.ndarray,
    #     R: jnp.ndarray,
    #     cell: jnp.ndarray,
    #     num_batch: int,
    #     batch_seg: jnp.ndarray,
    #     eps: float = 1e-8,
    # ) -> jnp.ndarray:
    #     box_length = jnp.diagonal(cell, axis1=-2, axis2=-1)
    #     k = self.two_pi * self.kmul / box_length[..., None]
    #     k2 = jnp.sum(k * k, axis=-1)
    #     qg = jnp.exp(-0.25 * k2 / self.alpha2) / k2
    #     dot = jnp.sum(k[batch_seg] * R[..., None], axis=-1)
    #     q_real = jnp.zeros((num_batch, dot.shape[-1])).at[batch_seg].add(q[..., None] * jnp.cos(dot))
    #     q_imag = jnp.zeros((num_batch, dot.shape[-1])).at[batch_seg].add(q[..., None] * jnp.sin(dot))
    #     qf = q_real ** 2 + q_imag ** 2
    #     e_reciprocal = (
    #         self.two_pi / jnp.prod(box_length, axis=1) * jnp.sum(qf * qg, axis=-1)
    #     )
    #     q2 = q * q
    #     e_self = self.alpha * self.one_over_sqrtpi * q2
    #     w = q2 + eps
    #     wnorm = jnp.zeros(num_batch).at[batch_seg].add(w)
    #     w = w / wnorm.at[batch_seg]
    #     e_reciprocal = w * e_reciprocal.at[batch_seg]
    #     return self.ke * (e_reciprocal - e_self)

    # def _ewald(
    #     self,
    #     N: int,
    #     q: jnp.ndarray,
    #     R: jnp.ndarray,
    #     rij: jnp.ndarray,
    #     idx_i: jnp.ndarray,
    #     idx_j: jnp.ndarray,
    #     cell: jnp.ndarray,
    #     num_batch: int,
    #     batch_seg: jnp.ndarray,
    # ) -> jnp.ndarray:
    #     e_real = self._real_space(N, q, rij, idx_i, idx_j)
    #     e_reciprocal = self._reciprocal_space(q, R, cell, num_batch, batch_seg)
    #     return e_real + e_reciprocal

    # def _coulomb_original(
    #     self,
    #     N: int,
    #     q: jnp.ndarray,
    #     rij: jnp.ndarray,
    #     idx_i: jnp.ndarray,
    #     idx_j: jnp.ndarray,
    # ) -> jnp.ndarray:
    #     fac = self.kehalf * q[idx_i] * q[idx_j]
    #     f = switch_function(rij, self.cuton, self.cutoff)
    #     if self.lr_cutoff is None:
    #         coulomb = 1.0 / rij
    #         damped = 1.0 / (rij ** 2 + 1.0) ** (1.0 / 2.0)
    #     else:
    #         coulomb = lax.where(
    #             rij < self.lr_cutoff,
    #             1.0 / rij + rij / self.lr_cutoff2 - self.two_div_cut,
    #             jnp.zeros_like(rij),
    #         )
    #         damped = lax.where(
    #             rij < self.lr_cutoff,
    #             1.0 / (rij ** 2 + 1.0) ** (1.0 / 2.0)
    #             + rij * self.rcutconstant
    #             - self.cutconstant,
    #             jnp.zeros_like(rij),
    #         )
    #     pairwise = fac * (f * damped + (1 - f) * coulomb)
    #     return jnp.zeros(N).at[idx_i].add(pairwise)
    

    # @partial(jit, static_argnums=1)
    # def _compute_num_pairs(self, num_nodes):
    #     return jnp.sum(num_nodes * (num_nodes - 1) / 2, dtype=jnp.int32)
    # @partial(jit, static_argnums=1)
    # def triu_indices(self, n: int, k: int = 0):
    #     return _triu_indices(n, k=k)
    
    @nn.compact
    def __call__(self, inputs: Dict, *args, **kwargs) -> jnp.ndarray:  
        node_mask = inputs['node_mask']  # (num_nodes)
        graph_mask = inputs['graph_mask']  # (num_graphs)
        num_nodes = len(node_mask)
        partial_charges = self.partial_charges(inputs)['partial_charges'] # shape: (num_nodes)
        d_ij_all = inputs['d_ij_all']  # shape: (num_pairs+1)
        cell = inputs.get('cell')
        batch_segments = inputs['batch_segments']  # (num_nodes)
        batch_segments_pairs = inputs['batch_segments_pairs']  # (num_pairs)
        i_pairs = inputs['i_pairs']
        j_pairs = inputs['j_pairs']

        # hirshfeld_ratios = self.hirshfeld_ratios(inputs)['hirshfeld_ratios']
        # hirshfeld_ratios = jnp.clip(jnp.abs(hirshfeld_ratios), 0.5, 1.1)

        # # Getting atomic numbers (needed to link to the free-atom reference values)
        # atomic_numbers = inputs['atomic_numbers']  # (num_nodes)

        # # Calculate alpha_ij and C6_ij using mixing rules
        # alpha_ij, C6_ij = mixing_rules(atomic_numbers, i_pairs, j_pairs, hirshfeld_ratios)
        
        # # Use cubic fit for sigma
        # sigma_ij = sigma_cubic_fit(alpha_ij)
    
        # atomic_electrostatic_energy_ij = _coulomb(partial_charges, d_ij_all, i_pairs, j_pairs, self.kehalf, self.cuton, self.cutoff)
        # atomic_electrostatic_energy_ij = _coulomb_erf(partial_charges, d_ij_all, i_pairs, j_pairs, self.kehalf, sigma_ij)
        
        atomic_electrostatic_energy_ij = _coulomb_erf(partial_charges, d_ij_all, i_pairs, j_pairs, self.kehalf, 1.64) 
        #TODO: 1.64 comes from sigma_cubic_fit(4.5), Hydrogen polarizability, multiplied by jnp.sqrt(2) 
        #TODO: insert correct sigma_ij as defined above, check if hirshfeld_ratios are slowing the program

        atomic_electrostatic_energy = segment_sum(
                atomic_electrostatic_energy_ij,
                # segment_ids=batch_segments_pairs,
                segment_ids=i_pairs,
                num_segments=num_nodes
            )  # (num_graphs)

        atomic_electrostatic_energy = safe_scale(atomic_electrostatic_energy, node_mask)

        return dict(electrostatic_energy=atomic_electrostatic_energy)

    def reset_output_convention(self, output_convention):
        self.output_convention = output_convention

        # if self.use_ewald_summation:
        #     assert d_ij_all is not None
        #     assert cell is not None
        #     assert batch_segments is not None
        #     electrostatic_energy = self._ewald(num_nodes, partial_charges, positions, d_ij_all, idx_i_all, idx_j_all, cell, num_batch, batch_segments)
        #     # print('Calc electrostatic_energy use_ewald_summation', electrostatic_energy)
        #     return dict(electrostatic_energy=electrostatic_energy)
        # else:
        #     electrostatic_energy = self._coulomb(num_nodes, partial_charges, d_ij_all, idx_i_all, idx_j_all)
            
        #     # print('Calc electrostatic_energy _coulomb', electrostatic_energy)
        #     return dict(electrostatic_energy=electrostatic_energy)


# def Damp(n, R, gamma) -> jnp.ndarray:
#     # Computes the QDO damping function of the order 2n
#     f = 1
#     for k in range(n+1):
#         f += -jnp.exp(-gamma*R**2/2)* gamma**k * R**(2*k)/2**k/jax.scipy.special.factorial(k)
#     return f

@jax.jit
def Damp_n3(R, gamma) -> jnp.ndarray:
    return 1 - jnp.exp(-gamma*R**2/2) * (1 + gamma*R**2/2 + (gamma*R**2/2)**2/2/factorial(2))

@jax.jit
def Damp_n4(R, gamma) -> jnp.ndarray:
    return 1 - jnp.exp(-gamma*R**2/2) * (1 + gamma*R**2/2 + (gamma*R**2/2)**2/2/factorial(2) + (gamma*R**2/2)**3/2/factorial(3))

@jax.jit
def Damp_n5(R, gamma) -> jnp.ndarray:
    return 1 - jnp.exp(-gamma*R**2/2) * (1 + gamma*R**2/2 + (gamma*R**2/2)**2/2/factorial(2) + (gamma*R**2/2)**3/2/factorial(3) + (gamma*R**2/2)**4/24/factorial(4))


@jax.jit
def vdw_QDO_disp_damp(R, gamma, C6):
    #  Computing the vdW-QDO dispersion energy and returning it in eV
    C8 = 5/gamma*C6
    C10 = 245/8/gamma**2*C6
    f6 = Damp_n3(R, gamma)
    f8 = Damp_n4(R, gamma)
    f10 = Damp_n5(R, gamma)
    # V1 = -f6*C6/R**6
    # V2 = -f6*C6/R**6 - f8*C8/R**8
    V3 = -f6*C6/R**6 - f8*C8/R**8 - f10*C10/R**10
    return V3*Hartree

@jax.jit
def mixing_rules(
    atomic_numbers: jnp.ndarray,
    idx_i: jnp.ndarray,
    idx_j: jnp.ndarray,
    hirshfeld_ratios: jnp.ndarray,
) -> jnp.ndarray:
    
    atomic_number_i = atomic_numbers[idx_i]-1
    atomic_number_j = atomic_numbers[idx_j]-1
    hirshefld_ratio_i = hirshfeld_ratios[idx_i]
    hirshefld_ratio_j = hirshfeld_ratios[idx_j]

    alpha_i = alphas[atomic_number_i] * hirshefld_ratio_i
    C6_i = C6_coef[atomic_number_i] * hirshefld_ratio_i**2
    alpha_j = alphas[atomic_number_j] * hirshefld_ratio_j
    C6_j = C6_coef[atomic_number_j] * hirshefld_ratio_j**2

    alpha_ij = (alpha_i + alpha_j) / 2
    C6_ij = 2 * C6_i * C6_j * alpha_j * alpha_i / (alpha_i**2 * C6_j + alpha_j**2 * C6_i)

    return alpha_ij, C6_ij

@jax.jit
def gamma_cubic_fit(alpha):
    vdW_radius = fine_structure**(-4/21)*alpha**(1/7) 
    b0 = -0.00433008
    b1 = 0.24428889
    b2 = 0.04125273
    b3 = -0.00078893
    sigma = b3*vdW_radius**3 + b2*vdW_radius**2 + b1*vdW_radius + b0
    gamma = 1/2/sigma**2
    return gamma#, sigma*jnp.sqrt(2)

class DispersionEnergySparse(BaseSubModule):
    prop_keys: Dict
    hirshfeld_ratios: Optional[Any] = None
    activation_fn: Callable[[Any], Any] = lambda u: u
    regression_dim: int = None
    module_name = 'dispersion_energy'
    output_is_zero_at_init: bool = True

    def setup(self):
        if self.output_is_zero_at_init:
            self.kernel_init = nn.initializers.zeros_init()
        else:
            self.kernel_init = nn.initializers.lecun_normal()
    
    @nn.compact
    def __call__(self, inputs: Dict, *args, **kwargs) -> jnp.ndarray:  
        node_mask = inputs['node_mask']  # (num_nodes)
        graph_mask = inputs['graph_mask']  # (num_graphs)
        # num_graphs = len(graph_mask)
        num_nodes = len(node_mask)
        d_ij_all = inputs['d_ij_all']  # shape: (num_pairs+1)
        # batch_segments = inputs['batch_segments']  # (num_nodes)
        # batch_segments_pairs = inputs['batch_segments_pairs']  # (num_pairs)
        pair_mask = inputs['pair_mask']  # (num_pairs)
        # input_convention: str = 'positions'
        # positions = inputs['positions'] # (num_nodes, 3)
        # total_charge = inputs['total_charge'] # (num_graphs)
        i_pairs = inputs['i_pairs']
        j_pairs = inputs['j_pairs']
        pair_mask = inputs['pair_mask']
        # num_pairs = len(pair_mask)
        # cell = inputs.get('cell')  # shape: (num_graphs, 3, 3)
        # cell_offsets = inputs.get('cell_offset')  # shape: (num_pairs, 3)

        hirshfeld_ratios = self.hirshfeld_ratios(inputs)['hirshfeld_ratios']
        mask_hirsh = hirshfeld_ratios != 7
        hirshfeld_ratios = jnp.clip(jnp.abs(hirshfeld_ratios), 0.5, 1.1)

        # Getting atomic numbers (needed to link to the free-atom reference values)
        atomic_numbers = inputs['atomic_numbers']  # (num_nodes)
        

        # Calculate alpha_ij and C6_ij using mixing rules
        alpha_ij, C6_ij = mixing_rules(atomic_numbers, i_pairs, j_pairs, hirshfeld_ratios)
        
        # Use cubic fit for gamma
        gamma_ij = gamma_cubic_fit(alpha_ij)

        # Getting dispersion energy, positions are converted to to a.u.
        dispersion_energy_ij = vdw_QDO_disp_damp(d_ij_all / Bohr, gamma_ij, C6_ij)

        dispersion_energy_ij = safe_scale(dispersion_energy_ij, pair_mask)

        atomic_dispersion_energy = segment_sum(
                dispersion_energy_ij,
                segment_ids=i_pairs,
                num_segments=num_nodes
            )  # (num_graphs)
        return dict(dispersion_energy=atomic_dispersion_energy)
    
    def reset_output_convention(self, output_convention):
        self.output_convention = output_convention