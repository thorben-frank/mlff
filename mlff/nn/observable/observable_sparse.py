import jax.numpy as jnp
import flax.linen as nn
from jax.ops import segment_sum
from typing import Any, Callable, Dict
from mlff.nn.base.sub_module import BaseSubModule
from jax.scipy.special import erfc
from typing import Optional
import sys

def _switch_component(x: jnp.ndarray, ones: jnp.ndarray, zeros: jnp.ndarray) -> jnp.ndarray:
    """ Component of the switch function, only for internal use. """
    x_ = jnp.where(x <= 0, ones, x)  # prevent nan in backprop
    return jnp.where(x <= 0, zeros, jnp.exp(-ones / x_))


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

        atomic_energy = jnp.where(node_mask, atomic_energy, jnp.asarray(0., dtype=atomic_energy.dtype))  # (num_nodes)

        if self.zbl_repulsion:
            raise NotImplementedError('ZBL Repulsion for sparse model not implemented yet.')

        if self.output_convention == 'per_structure':
            energy = segment_sum(
                atomic_energy,
                segment_ids=batch_segments,
                num_segments=num_graphs
            )  # (num_graphs)
            energy = jnp.where(graph_mask, energy, jnp.asarray(0., dtype=energy.dtype))

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
        center_of_mass = inputs['center_of_mass'] # (num_graphs, 3)

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
        x_q = jnp.where(node_mask, x_ + q_, jnp.asarray(0., dtype=x_.dtype))  # (num_nodes)
        # x_q = safe_scale(x_ + q_, scale=point_mask)  # shape: (n)

        total_charge_predicted = segment_sum(
            x_q,
            segment_ids=batch_segments,
            num_segments=num_graphs
        )  # (num_graphs)

        _, number_of_atoms_in_molecule = jnp.unique(batch_segments, return_counts = True, size=num_graphs)

        charge_conservation = (1 / number_of_atoms_in_molecule) * (total_charge - total_charge_predicted)
        partial_charges = x_q + jnp.repeat(charge_conservation, number_of_atoms_in_molecule, total_repeat_length = num_nodes)   # shape: (num_nodes)
        
        #Shift origin to center of mass to get consistent dipole moment for charged molecules
        center_of_mass_expanded = jnp.repeat(center_of_mass, number_of_atoms_in_molecule, axis = 0, total_repeat_length = num_nodes) # shape: (num_nodes, 3)
        positions_shifted = positions - center_of_mass_expanded

        #mu = positions * charges / (1e-11 / c / e)  # [num_nodes, 3]
        mu_i = positions_shifted * partial_charges[:, None] #(512,3) * (512,)

        dipole = segment_sum(
            mu_i,
            segment_ids=batch_segments,
            num_segments=num_graphs
        )  # (num_graphs, 3)
        dipole = jnp.linalg.norm(dipole, axis = 1)  # (num_graphs)
        dipole = jnp.where(graph_mask, dipole, jnp.asarray(0., dtype=dipole.dtype))

        return dict(dipole=dipole)
    
    # def reset_output_convention(self, output_convention):
    #     self.output_convention = output_convention

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {'output_is_zero_at_init': self.output_is_zero_at_init,
                                   'prop_keys': self.prop_keys}
                                   #'zmax': self.zmax,
                                #    'output_convention': self.output_convention,
                                #    'zbl_repulsion': self.zbl_repulsion,
                                #    'zbl_repulsion_shift': self.zbl_repulsion_shift,           
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
        # point_mask = inputs['point_mask']
        x = inputs['x']  # (num_nodes, num_features)
        node_mask = inputs['node_mask']  # (num_nodes)
        graph_mask = inputs['graph_mask']  # (num_graphs)
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
        q_x_k = jnp.where(node_mask, qk, jnp.asarray(0., dtype=k.dtype))

        v_eff = v_shift + q_x_k  # shape: (n)
        v_eff = jnp.where(node_mask, v_eff, jnp.asarray(0., dtype=v_eff.dtype))

        return dict(hirshfeld_ratios=v_eff)
    
class DipoleVecSparse(BaseSubModule):
    prop_keys: Dict
    regression_dim: int = None
    activation_fn: Callable[[Any], Any] = lambda u: u
    output_is_zero_at_init: bool = True
    module_name: str = 'dipole_vec'

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
        graph_mask_expanded = inputs['graph_mask_expanded']
        positions = inputs['positions'] # (num_nodes, 3)
        total_charge = inputs['total_charge'] # (num_graphs)
        center_of_mass = inputs['center_of_mass'] # (num_graphs, 3)

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
        x_q = jnp.where(node_mask, x_ + q_, jnp.asarray(0., dtype=x_.dtype))  # (num_nodes)
        # x_q = safe_scale(x_ + q_, scale=point_mask)  # shape: (n)

        total_charge_predicted = segment_sum(
            x_q,
            segment_ids=batch_segments,
            num_segments=num_graphs
        )  # (num_graphs)

        _, number_of_atoms_in_molecule = jnp.unique(batch_segments, return_counts = True, size=num_graphs)

        charge_conservation = (1 / number_of_atoms_in_molecule) * (total_charge - total_charge_predicted)
        partial_charges = x_q + jnp.repeat(charge_conservation, number_of_atoms_in_molecule, total_repeat_length = num_nodes)   # shape: (num_nodes)
        #TODO: Check whether partial charges make sense

        #Shift origin to center of mass to get consistent dipole moment for charged molecules
        center_of_mass_expanded = jnp.repeat(center_of_mass, number_of_atoms_in_molecule, axis = 0, total_repeat_length = num_nodes) # shape: (num_nodes, 3)
        positions_shifted = positions - center_of_mass_expanded
        
        #mu = positions * charges / (1e-11 / c / e)  # [num_nodes, 3]
        mu_i = positions_shifted * partial_charges[:, None] #(512,3) * (512,)

        dipole = segment_sum(
            mu_i,
            segment_ids=batch_segments,
            num_segments=num_graphs
        )  # (num_graphs, 3)

        dipole_vec = jnp.where(graph_mask_expanded, dipole, jnp.asarray(0., dtype=dipole.dtype))

        return dict(dipole_vec=dipole_vec)
    

# if use_electrostatics:
# self.electrostatic_energy = ElectrostaticEnergy(
#     cuton=0.25 * self.cutoff,
#     cutoff=0.75 * self.cutoff,
#     lr_cutoff=self.lr_cutoff,
# )
    
#     # optimization when lr_cutoff is used
# if self.lr_cutoff is not None and (
#     self.use_electrostatics or self.use_d4_dispersion
# ):
#     mask = rij < self.lr_cutoff  # select all entries below lr_cutoff
#     rij = rij[mask]
#     idx_i = idx_i[mask]
#     idx_j = idx_j[mask]

# # compute electrostatic contributions
# if self.use_electrostatics:
#     ea_ele = self.electrostatic_energy(
#         N, qa, rij, idx_i, idx_j, R, cell, num_batch, batch_seg
#     )
# else:
#     ea_ele = ea.new_zeros(N)
        
class ElectrostaticEnergy(BaseSubModule):
    prop_keys: Dict
    # zmax: int = 118
    # regression_dim: int = None
    # activation_fn: Callable[[Any], Any] = lambda u: u
    # learn_atomic_type_scales: bool = False
    # learn_atomic_type_shifts: bool = False
    # zbl_repulsion: bool = False
    # zbl_repulsion_shift: float = 0.
    # output_is_zero_at_init: bool = True
    # output_convention: str = 'per_structure'
    module_name: str = 'electrostatic_energy'

    ke: float = 14.399645351950548,
    cuton: float = 0.0,
    cutoff: float = 1.0,
    lr_cutoff: Optional[float] = None,    

    kehalf = ke / 2
    use_ewald_summation = False
    alpha = 0.0
    alpha2 = 0.0
    two_pi = 2.0 * jnp.pi
    one_over_sqrtpi = 1 / jnp.sqrt(jnp.pi)
    kmul = jnp.array([])

    def set_lr_cutoff(self, lr_cutoff: Optional[float] = None) -> None:
        """ Change the long range cutoff. """
        self.lr_cutoff = lr_cutoff
        if self.lr_cutoff is not None:
            self.lr_cutoff2 = lr_cutoff ** 2
            self.two_div_cut = 2.0 / lr_cutoff
            self.rcutconstant = lr_cutoff / (lr_cutoff ** 2 + 1.0) ** (3.0 / 2.0)
            self.cutconstant = (2 * lr_cutoff ** 2 + 1.0) / (lr_cutoff ** 2 + 1.0) ** (
                3.0 / 2.0
            )
        else:
            self.lr_cutoff2 = None
            self.two_div_cut = None
            self.rcutconstant = None
            self.cutconstant = None

    def set_kmax(self, Nxmax: int, Nymax: int, Nzmax: int) -> None:
        """ Set integer reciprocal space cutoff for Ewald summation """
        kx = jnp.arange(0, Nxmax + 1)
        kx = jnp.concatenate([kx, -kx[1:]])
        ky = jnp.arange(0, Nymax + 1)
        ky = jnp.concatenate([ky, -ky[1:]])
        kz = jnp.arange(0, Nzmax + 1)
        kz = jnp.concatenate([kz, -kz[1:]])
        kmul = jnp.array(jnp.meshgrid(kx, ky, kz)).reshape(3, -1).T[1:]  # 0th entry is 0 0 0
        kmax = max(max(Nxmax, Nymax), Nzmax)
        self.kmul = kmul[jnp.sum(kmul ** 2, axis=-1) <= kmax ** 2]

    def set_alpha(self, alpha: Optional[float] = None) -> None:
        """ Set real space damping parameter for Ewald summation """
        if alpha is None:  # automatically determine alpha
            alpha = 4.0 / self.cutoff + 1e-3
        self.alpha = alpha
        self.alpha2 = alpha ** 2
        self.two_pi = 2.0 * jnp.pi
        self.one_over_sqrtpi = 1 / jnp.sqrt(jnp.pi)
        # print a warning if alpha is so small that the reciprocal space sum
        # might "leak" into the damped part of the real space coulomb interaction
        if alpha * self.cutoff < 4.0:  # erfc(4.0) ~ 1e-8
            print(
                "Warning: Damping parameter alpha is",
                alpha,
                "but probably should be at least",
                4.0 / self.cutoff,
            )

    def _real_space(
        self,
        N: int,
        q: jnp.ndarray,
        rij: jnp.ndarray,
        idx_i: jnp.ndarray,
        idx_j: jnp.ndarray,
    ) -> jnp.ndarray:
        if q.device.type == "cpu":  # indexing is faster on CPUs
            fac = self.kehalf * q[idx_i] * q[idx_j]
        else:  # gathering is faster on GPUs
            fac = self.kehalf * jnp.take(q, idx_i) * jnp.take(q, idx_j)
        f = switch_function(rij, self.cuton, self.cutoff)
        coulomb = 1.0 / rij
        damped = 1.0 / (rij ** 2 + 1.0) ** (1.0 / 2.0)
        pairwise = fac * (f * damped + (1 - f) * coulomb) * erfc(self.alpha * rij)
        return jnp.zeros(N).at[idx_i].add(pairwise)

    def _reciprocal_space(
        self,
        q: jnp.ndarray,
        R: jnp.ndarray,
        cell: jnp.ndarray,
        num_batch: int,
        batch_seg: jnp.ndarray,
        eps: float = 1e-8,
    ) -> jnp.ndarray:
        # calculate k-space vectors
        box_length = jnp.diagonal(cell, axis1=-2, axis2=-1)
        k = self.two_pi * self.kmul[jnp.newaxis, :] / box_length[..., jnp.newaxis]
        # gaussian charge density
        k2 = jnp.sum(k * k, axis=-1)  # squared length of k-vectors
        qg = jnp.exp(-0.25 * k2 / self.alpha2) / k2
        # fourier charge density
        if q.device.type == "cpu":  # indexing is faster on CPUs
            dot = jnp.sum(k[batch_seg] * R[..., jnp.newaxis], axis=-1)
        else:  # gathering is faster on GPUs
            b = batch_seg.reshape(-1, 1, 1).repeat(1, k.shape[-2], k.shape[-1])
            dot = jnp.sum(jnp.take(k, b, axis=0) * R[..., jnp.newaxis], axis=-1)
        q_real = jnp.zeros((num_batch, dot.shape[-1])).at[batch_seg].add(q[..., jnp.newaxis] * jnp.cos(dot))
        q_imag = jnp.zeros((num_batch, dot.shape[-1])).at[batch_seg].add(q[..., jnp.newaxis] * jnp.sin(dot))
        qf = q_real ** 2 + q_imag ** 2
        # reciprocal energy
        e_reciprocal = (
            self.two_pi / jnp.prod(box_length, axis=1) * jnp.sum(qf * qg, axis=-1)
        )
        # self interaction correction
        q2 = q * q
        e_self = self.alpha * self.one_over_sqrtpi * q2
        # spread reciprocal energy over atoms (to get an atomic contributions)
        w = q2 + eps  # epsilon is added to prevent division by zero
        wnorm = jnp.zeros(num_batch).at[batch_seg].add(w)
        if w.device.type == "cpu":  # indexing is faster on CPUs
            w = w / wnorm[batch_seg]
            e_reciprocal = w * e_reciprocal[batch_seg]
        else:  # gathering is faster on GPUs
            w = w / jnp.take(wnorm, batch_seg, axis=0)
            e_reciprocal = w * jnp.take(e_reciprocal, batch_seg, axis=0)
        return self.ke * (e_reciprocal - e_self)

    def _ewald(
        self,
        N: int,
        q: jnp.ndarray,
        R: jnp.ndarray,
        rij: jnp.ndarray,
        idx_i: jnp.ndarray,
        idx_j: jnp.ndarray,
        cell: jnp.ndarray,
        num_batch: int,
        batch_seg: jnp.ndarray,
    ) -> jnp.ndarray:
        e_real = self._real_space(N, q, rij, idx_i, idx_j)
        e_reciprocal = self._reciprocal_space(q, R, cell, num_batch, batch_seg)
        return e_real + e_reciprocal
    
    def _coulomb(
        self,
        N: int,
        q: jnp.ndarray,
        rij: jnp.ndarray,
        idx_i: jnp.ndarray,
        idx_j: jnp.ndarray,
    ) -> jnp.ndarray:
        if q.device.type == "cpu":  # indexing is faster on CPUs
            fac = self.kehalf * q[idx_i] * q[idx_j]
        else:  # gathering is faster on GPUs
            fac = self.kehalf * jnp.take(q, idx_i) * jnp.take(q, idx_j)
        f = switch_function(rij, self.cuton, self.cutoff)
        if self.lr_cutoff is None:
            coulomb = 1.0 / rij
            damped = 1.0 / (rij ** 2 + 1.0) ** (1.0 / 2.0)
        else:
            coulomb = jnp.where(
                rij < self.lr_cutoff,
                1.0 / rij + rij / self.lr_cutoff2 - self.two_div_cut,
                jnp.zeros_like(rij),
            )
            damped = jnp.where(
                rij < self.lr_cutoff,
                1.0 / (rij ** 2 + 1.0) ** (1.0 / 2.0)
                + rij * self.rcutconstant
                - self.cutconstant,
                jnp.zeros_like(rij),
            )
        pairwise = fac * (f * damped + (1 - f) * coulomb)
        return jnp.zeros(N).at[idx_i].add(pairwise)
    
        # N: Number of atoms.
        # P: Number of atom pairs.
        # B: Batch size (number of different molecules).
    def forward(
        self,
        N: int, #number of atoms
        q: jnp.ndarray, #qa - Atomic partial charges [N]
        rij: jnp.ndarray, #Pairwise interatomic distances [P]
        idx_i: jnp.ndarray, #(LongTensor [P]): Index of atom i for all atomic pairs ij. Each pair must be specified as both ij and ji.
        idx_j: jnp.ndarray,
        R: Optional[jnp.ndarray] = None, #Cartesian coordinates (x,y,z) of atoms (FloatTensor [N, 3])
        cell: Optional[jnp.ndarray] = None, #FloatTensor [B, 3, 3] or None
        num_batch: int = 1, #Batch size (number of different molecules)
        batch_seg: Optional[jnp.ndarray] = None,
                #    batch_seg (LongTensor [N]):
                # Index for each atom that specifies to which molecule in the
                # batch it belongs. For example, when predicting a H2O and a CH4
                # molecule, batch_seg would be [0, 0, 0, 1, 1, 1, 1, 1] to
                # indicate that the first three atoms belong to the first molecule
                # and the last five atoms to the second molecule.
    ) -> jnp.ndarray:
        if self.use_ewald_summation:
            assert R is not None
            assert cell is not None
            assert batch_seg is not None
            return self._ewald(N, q, R, rij, idx_i, idx_j, cell, num_batch, batch_seg)
        else:
            return self._coulomb(N, q, rij, idx_i, idx_j)

    #N - number of atoms... in batch or where
    #qa - need to pass from DipoleVecSparse
    #rij - need to pass from ase dataloader
    #idx_i - need to pass from ase dataloader, senders
    #idx_j - need to pass from ase dataloader, receivers
    #R - done
    #cell - none for now
    #num_batch - ?
    #batch_seg - ?
        
        #    def energy_fn(params,
        #           positions: jnp.ndarray,
        #           atomic_numbers: jnp.ndarray,
        #           idx_i: jnp.ndarray,
        #           idx_j: jnp.ndarray,
        #           cell: jnp.ndarray = None,
        #           cell_offset: jnp.ndarray = None,
        #           batch_segments: jnp.ndarray = None,
        #           node_mask: jnp.ndarray = None,
        #           graph_mask: jnp.ndarray = None,
        #           graph_mask_expanded: jnp.ndarray = None,
        #           total_charge: jnp.ndarray = None,
        #           center_of_mass: jnp.ndarray = None):

    @nn.compact
    def __call__(self, inputs: Dict, *args, **kwargs):
        graph_mask = inputs['graph_mask']  # (num_graphs)
        N = len(graph_mask) #not sure
        qa = inputs['qa'] #need to pass it somehow
        rij = inputs['rij']
        idx_i = inputs['idx_i']
        idx_j = inputs['idx_j']
        R = inputs['positions']
        cell = inputs['cell']
        # num_batch = inputs['']
        # batch_seg = inputs['batch_segments']  # (num_nodes)

        if self.use_ewald_summation:
            assert R is not None
            assert cell is not None
            assert batch_seg is not None
            return self._ewald(N, q, R, rij, idx_i, idx_j, cell, num_batch, batch_seg)
        else:
            return self._coulomb(N, q, rij, idx_i, idx_j)   

        #need to return coulomb energy for the batch
        
            
        # if self.output_convention == 'per_atom':
        #     return segment_sum(e_rep_edge, segment_ids=idx_i, num_segments=len(z))[:, None]  # shape: (n,1)
        # elif self.output_convention == 'per_structure':
        #     return e_rep_edge.sum(axis=0)  # shape: (1) 

  
    # @nn.compact
    # def __call__(self, inputs: Dict, *args, **kwargs):
    #     x = inputs['x']  # (num_nodes, num_features)
    #     atomic_numbers = inputs['atomic_numbers']  # (num_nodes)
    #     batch_segments = inputs['batch_segments']  # (num_nodes)
    #     node_mask = inputs['node_mask']  # (num_nodes)
    #     graph_mask = inputs['graph_mask']  # (num_graphs)

    #     num_graphs = len(graph_mask)

    #     atomic_energy = func(x)  # (num_nodes)

    #     atomic_energy = jnp.where(node_mask, atomic_energy, jnp.asarray(0., dtype=atomic_energy.dtype))  # (num_nodes)
        
    #     energy = segment_sum(
    #         atomic_energy,
    #         segment_ids=batch_segments,
    #         num_segments=num_graphs
    #     )  # (num_graphs)
    #     energy = jnp.where(graph_mask, energy, jnp.asarray(0., dtype=energy.dtype))

    #     return dict(energy=energy)
