import jax
import jax.numpy as jnp
import flax.linen as nn
from jax.ops import segment_sum
from typing import Any, Callable, Dict
from mlff.nn.base.sub_module import BaseSubModule
from typing import Optional
from mlff.cutoff_function import add_cell_offsets_sparse
from mlff.masking.mask import safe_mask, safe_norm
from ase.units import Bohr, Hartree
from ase.units import alpha as fine_structure
from mlff.nn.observable.dispersion_ref_data import alphas, C6_coef
from jax.scipy.special import factorial
from mlff.masking.mask import safe_scale
from mlff.nn.activation_function.activation_function import silu, softplus_inverse, softplus
from jax.nn.initializers import constant

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

@jax.jit
def sigma(x):
    return safe_mask(x > 0, fn=lambda u: jnp.exp(-1. / u), operand=x, placeholder=0)

@jax.jit
def switching_fn(x, x_on, x_off):
    c = (x - x_on) / (x_off - x_on)
    return sigma(1 - c) / (sigma(1 - c) + sigma(c))

@jax.jit
def Damp_n3(z) -> jnp.ndarray:
    return 1 - jnp.exp(-z) * (1 + z + z**2/factorial(2) + z**3/factorial(3))

@jax.jit
def Damp_n4(z) -> jnp.ndarray:
    return 1 - jnp.exp(-z) * (1 + z + z**2/factorial(2) + z**3/factorial(3)+z**4/factorial(4))

@jax.jit
def Damp_n5(z) -> jnp.ndarray:
    return 1 - jnp.exp(-z) * (1 + z + z**2/factorial(2) + z**3/factorial(3)+z**4/factorial(4)+z**5/factorial(5))


@jax.jit
def vdw_QDO_disp_damp(R, gamma, C6):
    #  Compute the vdW-QDO dispersion energy (in eV)
    z = gamma*R**2/2
    C8 = 5/gamma*C6
    C10 = 245/8/gamma**2*C6
    f6 = Damp_n3(z)
    f8 = Damp_n4(z)
    f10 = Damp_n5(z)
    V3 = -f6*C6/R**6 - f8*C8/R**8 - f10*C10/R**10
    V3_1 = jnp.multiply(V3, 0.5)
    return V3_1*Hartree

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
    return gamma

@jax.jit
def _coulomb_erf(q: jnp.ndarray, rij: jnp.ndarray, 
             idx_i: jnp.ndarray, idx_j: jnp.ndarray,
             kehalf: float, sigma: float
) -> jnp.ndarray:
    """ Pairwise Coulomb interaction with erf damping """
    pairwise = kehalf * q[idx_i] * q[idx_j] * jax.scipy.special.erf(rij/sigma)/rij
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
def _coulomb_pme(q: jnp.ndarray, positions : jnp.ndarray, cell: jnp.ndarray, 
             ngrid: jnp.ndarray, alpha: float, frequency: jnp.ndarray,
) -> jnp.ndarray:
    """ Pairwise Coulomb interaction with erf damping plus PME"""
    # Necesito hacer el grid, se supone q debe alocarse solo una vez y luego solamente se actualiza, hablar con adil

    
    @partial(jax.jit, static_argnums=(3,))
    def map_charges_to_grid(positions, q, icell, ngrid):
        """Smears charges over a grid of specified dimensions."""
        # Jax-md implementation https://github.com/jax-md/jax-md/blob/main/jax_md/_energy/electrostatics.py
        Q = ngrid
        N = positions.shape[0]

        @partial(jnp.vectorize, signature='(),()->(p)')
        def grid_position(u, K):
            grid = jnp.floor(u).astype(jnp.int32)
            grid = jnp.arange(0, 4) + grid
            return jnp.mod(grid, K)

        @partial(jnp.vectorize, signature='(d),()->(p,p,p,d),(p,p,p)')
        def map_particle_to_grid(positions, charge):
            u = raw_transform(icell, positions) * grid_dimensions
            w = u - jnp.floor(u)
            coeffs = optimized_bspline_4(w)

            grid_pos = grid_position(u, grid_dimensions)

            accum = charge * (coeffs[0, :, None, None] *
                                coeffs[1, None, :, None] *
                                coeffs[2, None, None, :])
            grid_pos = jnp.concatenate(
                (jnp.broadcast_to(grid_pos[[0], :, None, None], (1, 4, 4, 4)),
                    jnp.broadcast_to(grid_pos[[1], None, :, None], (1, 4, 4, 4)),
                    jnp.broadcast_to(grid_pos[[2], None, None, :], (1, 4, 4, 4))), axis=0)
            grid_pos = jnp.transpose(grid_pos, (1, 2, 3, 0))

            return grid_pos, accum

        gp, ac = map_particle_to_grid(positions, q)
        gp = jnp.reshape(gp, (-1, 3))
        ac = jnp.reshape(ac, (-1,))

        return Q.at[gp[:, 0], gp[:, 1], gp[:, 2]].add(ac)
    
    def _get_free_indices(n: int) -> str:
        return ''.join([chr(ord('a') + i) for i in range(n)])

    def raw_transform(box, R) -> Array:
        """Apply an affine transformation to positions.

        See `periodic_general` for a description of the semantics of `box`.

        Args:
            box: An affine transformation described in `periodic_general`.
            R: Array of positions. Should have  shape `(..., spatial_dimension)`.

        Returns:
            A transformed array positions of shape `(..., spatial_dimension)`.
        """
        free_indices = _get_free_indices(R.ndim - 1)
        left_indices = free_indices + 'j'
        right_indices = free_indices + 'i'
        return jnp.einsum(f'ij,{left_indices}->{right_indices}', box, R)



    @partial(jnp.vectorize, signature='()->(p)')
    def optimized_bspline_4(w):
        coeffs = jnp.zeros((4,))

        coeffs = coeffs.at[2].set(0.5 * w * w)
        coeffs = coeffs.at[0].set(0.5 * (1.0-w) * (1.0-w))
        coeffs = coeffs.at[1].set(1.0 - coeffs[0] - coeffs[2])

        coeffs = coeffs.at[3].set(w * coeffs[2] / 3.0)
        coeffs = coeffs.at[2].set(((1.0 + w) * coeffs[1] + (3.0 - w) * coeffs[2])/3.0)
        coeffs = coeffs.at[0].set((1.0 - w) * coeffs[0] / 3.0)
        coeffs = coeffs.at[1].set(1.0 - coeffs[0] - coeffs[2] - coeffs[3])

        return coeffs
    
    @partial(jnp.vectorize, signature='()->()')
    def b(m, n=4):
        assert(n == 4)
        k = jnp.arange(n - 1)
        M = optimized_bspline_4(1.0)[1:][::-1]
        prefix = jnp.exp(2 * jnp.pi * 1j * (n - 1) * m)
        return prefix / jnp.sum(M * jnp.exp(2 * jnp.pi * 1j * m * k))


    def B(mx, my, mz, n=4):
        """Compute the B factors from Essmann et al. equation 4.7."""
        b_x = b(mx)
        b_y = b(my)
        b_z = b(mz)
        return jnp.abs(b_x)**2 * jnp.abs(b_y)**2 * jnp.abs(b_z)**2
    

    icell = jnp.linalg.inv(cell)
    grid_dimensions = jnp.array(ngrid.shape)
    grid = map_charges_to_grid(positions, q, icell, ngrid)
    Fgrid = jnp.fft.fftn(grid)
    mx, my, mz = frequency

    m = (icell[None, None, None, 0] * mx[:, :, :, None] * grid_dimensions[0] +
        icell[None, None, None, 1] * my[:, :, :, None] * grid_dimensions[1] +
        icell[None, None, None, 2] * mz[:, :, :, None] * grid_dimensions[2])
    m_2 = jnp.sum(m**2, axis=-1)
    V = jnp.linalg.det(cell)
    mask = m_2 != 0

    exp_m = 1 / (2 * jnp.pi * V) * jnp.exp(-jnp.pi**2 * m_2 / alpha**2) / m_2
    return jnp.sum(mask * exp_m * B(mx, my, mz) * jnp.abs(Fgrid)**2)


Array = Any

class EnergySparse(BaseSubModule):
    prop_keys: Dict
    zmax: int = 118
    regression_dim: int = None
    activation_fn: Callable[[Any], Any] = lambda u: u
    learn_atomic_type_scales: bool = False
    learn_atomic_type_shifts: bool = False
    output_is_zero_at_init: bool = True
    output_convention: str = 'per_structure'
    module_name: str = 'energy'
    electrostatic_energy_bool: bool = False
    electrostatic_energy: Optional[Any] = None
    dispersion_energy_bool: bool = False
    dispersion_energy: Optional[Any] = None
    zbl_repulsion_bool: bool = False
    zbl_repulsion: Optional[Any] = None

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

        if self.zbl_repulsion_bool:
            repulsion_energy = self.zbl_repulsion(inputs)['zbl_repulsion']
            atomic_energy += repulsion_energy
        
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
            # return dict(e_sum = jnp.sum(electrostatic_energy), 
            #             d_sum = jnp.sum(dispersion_energy), 
            #             r_sum = jnp.sum(repulsion_energy), 
            #             energy=energy, 
            #             atomic_energy=atomic_energy ,
            #             repulsion_energy=repulsion_energy, 
            #             electrostatic_energy=electrostatic_energy, 
            #             dispersion_energy=dispersion_energy)

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
                                   'zbl_repulsion_bool': self.zbl_repulsion_bool,
                                   'electrostatic_energy_bool': self.electrostatic_energy_bool,
                                   'dispersion_energy_bool': self.dispersion_energy_bool,
                                   'prop_keys': self.prop_keys}
                }      

class ZBLRepulsionSparse(BaseSubModule):
    """
    Ziegler-Biersack-Littmark repulsion.
    """
    prop_keys: Dict
    input_convention: str = 'positions'
    module_name: str = 'zbl_repulsion'
    a0: float = 0.5291772105638411
    ke: float = 14.399645351950548

    @nn.compact
    def __call__(self,
                inputs: Dict,
                *args,
                **kwargs) -> Dict[str, jnp.ndarray]:

        a1 = softplus(self.param('a1', constant(softplus_inverse(3.20000)), (1,)))  # shape: (1)
        a2 = softplus(self.param('a2', constant(softplus_inverse(0.94230)), (1,)))  # shape: (1)
        a3 = softplus(self.param('a3', constant(softplus_inverse(0.40280)), (1,)))  # shape: (1)
        a4 = softplus(self.param('a4', constant(softplus_inverse(0.20160)), (1,)))  # shape: (1)
        c1 = softplus(self.param('c1', constant(softplus_inverse(0.18180)), (1,)))  # shape: (1)
        c2 = softplus(self.param('c2', constant(softplus_inverse(0.50990)), (1,)))  # shape: (1)
        c3 = softplus(self.param('c3', constant(softplus_inverse(0.28020)), (1,)))  # shape: (1)
        c4 = softplus(self.param('c4', constant(softplus_inverse(0.02817)), (1,)))  # shape: (1)
        p = softplus(self.param('p', constant(softplus_inverse(0.23)), (1,)))  # shape: (1)
        d = softplus(self.param('d', constant(softplus_inverse(1 / (0.8854 * self.a0))), (1,)))  # shape: (1)

        c_sum = c1 + c2 + c3 + c4
        c1 = c1 / c_sum
        c2 = c2 / c_sum
        c3 = c3 / c_sum
        c4 = c4 / c_sum

        atomic_numbers = inputs['atomic_numbers']
        node_mask = inputs['node_mask']
        num_nodes = len(node_mask)
        phi_r_cut_ij = inputs['cut']
        idx_i = inputs['idx_i']
        idx_j = inputs['idx_j']

        z_i = atomic_numbers[idx_i]
        z_j = atomic_numbers[idx_j]

        d_ij = inputs['d_ij']

        z_d_ij = safe_mask(mask=d_ij != 0,
                           operand=d_ij,
                           fn=lambda u: z_i * z_j / u,
                           placeholder=0.
                           )

        x = self.ke * phi_r_cut_ij * z_d_ij

        rzd = d_ij * (jnp.power(z_i, p) + jnp.power(z_j, p)) * d
        y = c1 * jnp.exp(-a1 * rzd) + c2 * jnp.exp(-a2 * rzd) + c3 * jnp.exp(-a3 * rzd) + c4 * jnp.exp(-a4 * rzd)

        w = switching_fn(d_ij, x_on=0, x_off=1.5)

        e_rep_edge = w * x * y / jnp.asarray(2, dtype=d_ij.dtype)
        e_rep_edge = segment_sum(e_rep_edge, segment_ids=idx_i, num_segments=num_nodes)
        e_rep_edge = safe_scale(e_rep_edge, node_mask)

        return dict(zbl_repulsion=e_rep_edge)
        # if self.output_convention == 'per_atom':
        #     return dict(zbl_repulsion=e_rep_edge)
        # else:
        #     raise ValueError(f"{self.output_convention} is invalid argument for attribute `output_convention`.")
        
    def reset_output_convention(self, output_convention):
        self.output_convention = output_convention

class HirshfeldSparse(BaseSubModule):
    prop_keys: Dict
    regression_dim: int = None
    activation_fn: Callable[[Any], Any] = lambda u: u
    output_is_zero_at_init: bool = True
    module_name = 'hirshfeld_ratios'

    def setup(self):
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

        x_q = safe_scale(x_ + q_, node_mask)

        total_charge_predicted = segment_sum(
            x_q,
            segment_ids=batch_segments,
            num_segments=num_graphs
        )  # (num_graphs)

        _, number_of_atoms_in_molecule = jnp.unique(batch_segments, return_counts = True, size=num_graphs)

        charge_conservation = (1 / number_of_atoms_in_molecule) * (total_charge - total_charge_predicted)
        partial_charges = x_q + jnp.repeat(charge_conservation, number_of_atoms_in_molecule, total_repeat_length = num_nodes)   # shape: (num_nodes)
        
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

        mu_i = positions * partial_charges[:, None]

        dipole = segment_sum(
            mu_i,
            segment_ids=batch_segments,
            num_segments=num_graphs
        )  # (num_graphs, 3)

        dipole_vec = safe_scale(dipole, graph_mask_expanded)

        return dict(dipole_vec=dipole_vec)
        # return dict(dipole_vec=dipole_vec, 
        #             partial_charges=partial_charges)

    def reset_output_convention(self, output_convention):
        self.output_convention = output_convention

class ElectrostaticEnergySparse(BaseSubModule):
    prop_keys: Dict
    regression_dim: int = None
    activation_fn: Callable[[Any], Any] = lambda u: u
    output_is_zero_at_init: bool = True
    module_name: str = 'electrostatic_energy'
    input_convention: str = 'positions'
    partial_charges: Optional[Any] = None
    use_particle_mesh_ewald: bool = False
    kehalf: float = 14.399645351950548/2  #TODO: should we use ke or kehalf?
    electrostatic_energy_scale: float = 1.0
  
    @nn.compact
    def __call__(self, inputs: Dict, *args, **kwargs) -> jnp.ndarray:  
        node_mask = inputs['node_mask']  # (num_nodes)
        num_nodes = len(node_mask)
        partial_charges = self.partial_charges(inputs)['partial_charges']
        idx_i_lr = inputs['idx_i_lr']
        idx_j_lr = inputs['idx_j_lr']        
        d_ij_lr = inputs['d_ij_lr']
       
        atomic_electrostatic_energy_ij = _coulomb_erf(partial_charges, d_ij_lr, idx_i_lr, idx_j_lr, self.ke, self.electrostatic_energy_scale)

        atomic_electrostatic_energy = segment_sum(
                atomic_electrostatic_energy_ij,
                segment_ids=idx_i_lr,
                num_segments=num_nodes
            )  # (num_nodes)

        atomic_electrostatic_energy = safe_scale(atomic_electrostatic_energy, node_mask)

        ngrid = inputs['ngrid'] # Check if ngrid is not None. Temporary solution to check if use PME
        if ngrid:
            N = len(partial_charges)
            positions = inputs['positions']
            cell = inputs['cell']
            alpha = inputs['alpha']
            frequency = inputs['frequency']

            atomic_electrostatic_energy += _coulomb_pme(partial_charges, positions, cell, ngrid, alpha, frequency)/N

        return dict(electrostatic_energy=atomic_electrostatic_energy)

    def reset_output_convention(self, output_convention):
        self.output_convention = output_convention

class DispersionEnergySparse(BaseSubModule):
    prop_keys: Dict
    hirshfeld_ratios: Optional[Any] = None
    activation_fn: Callable[[Any], Any] = lambda u: u
    regression_dim: int = None
    module_name = 'dispersion_energy'
    input_convention: str = 'positions'
    output_is_zero_at_init: bool = True
    dispersion_energy_scale: float = 1.0

    def setup(self):
        if self.output_is_zero_at_init:
            self.kernel_init = nn.initializers.zeros_init()
        else:
            self.kernel_init = nn.initializers.lecun_normal()
    
    @nn.compact
    def __call__(self, inputs: Dict, *args, **kwargs) -> jnp.ndarray:  
        node_mask = inputs['node_mask']  # (num_nodes)
        num_nodes = len(node_mask)
        idx_i_lr = inputs['idx_i_lr']
        idx_j_lr = inputs['idx_j_lr']
        d_ij_lr = inputs['d_ij_lr']

        hirshfeld_ratios = self.hirshfeld_ratios(inputs)['hirshfeld_ratios']

        # Get atomic numbers (needed to link to the free-atom reference values)
        atomic_numbers = inputs['atomic_numbers']  # (num_nodes)
        
        # Calculate alpha_ij and C6_ij using mixing rules
        alpha_ij, C6_ij = mixing_rules(atomic_numbers, idx_i_lr, idx_j_lr, hirshfeld_ratios)
        
        # Use cubic fit for gamma
        gamma_ij = gamma_cubic_fit(alpha_ij)/self.dispersion_energy_scale

        # Get dispersion energy, positions are converted to to a.u.
        dispersion_energy_ij = vdw_QDO_disp_damp(d_ij_lr / Bohr, gamma_ij, C6_ij)

        atomic_dispersion_energy = segment_sum(
                dispersion_energy_ij,
                segment_ids=idx_i_lr,
                num_segments=num_nodes
            )  # (num_nodes)
        
        atomic_dispersion_energy = safe_scale(atomic_dispersion_energy, node_mask)

        return dict(dispersion_energy=atomic_dispersion_energy)
    
    def reset_output_convention(self, output_convention):
        self.output_convention = output_convention