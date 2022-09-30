import jax
import jax.numpy as jnp
import flax.linen as nn

from functools import partial
from typing import (Any, Dict, Sequence)
from jax.ops import segment_sum

from mlff.src.nn.base.sub_module import BaseSubModule
from mlff.src.masking.mask import safe_mask
from mlff.src.nn.mlp import MLP
from mlff.src.masking.mask import safe_scale
from mlff.src.nn.activation_function import silu
from mlff.src.basis_function.radial import get_rbf_fn
from mlff.src.cutoff_function.pbc import pbc_diff
from mlff.src.cutoff_function.radial import get_cutoff_fn
from mlff.src.sph_ops.spherical_harmonics import init_sph_fn


# TODO: write init_from_dict methods in order to improve backward compatibility. E.g. AtomTypeEmbed(**h)
# will only work as long as the properties of the class are exactly the ones equal to the ones in h. As soon
# as additional arguments appear in h. Maybe use something like kwargs to allow for extensions?


class GeometryEmbed(BaseSubModule):
    prop_keys: Dict
    degrees: Sequence[int]
    radial_basis_function: str
    n_rbf: int
    radial_cutoff_fn: str
    r_cut: float
    sphc: bool
    pbc: bool = False
    solid_harmonic: bool = False
    module_name: str = 'geometry_embed'

    def setup(self):
        self.atomic_position_key = self.prop_keys.get('atomic_position')
        self.atomic_type_key = self.prop_keys.get('atomic_type')
        if self.pbc:
            self.lattice_vector_key = self.prop_keys.get('lattice_vector')

        self.sph_fns = [init_sph_fn(y) for y in self.degrees]

        _rbf_fn = get_rbf_fn(self.radial_basis_function)
        self.rbf_fn = _rbf_fn(n_rbf=self.n_rbf, r_cut=self.r_cut)

        _cut_fn = get_cutoff_fn(self.radial_cutoff_fn)
        self.cut_fn = partial(_cut_fn, r_cut=self.r_cut)

    def __call__(self, inputs: Dict, *args, **kwargs):
        """
        Embed geometric information from the atomic positions and its neighboring atoms.

        Args:
            inputs (Dict):
                R (Array): atomic positions, shape: (n,3)
                idx_i (Array): index centering atom, shape: (n_pairs)
                idx_j (Array): index neighboring atom, shape: (n_pairs)
                pair_mask (Array): index based mask to exclude pairs that come from index padding, shape: (n_pairs)
            *args ():
            **kwargs ():

        Returns:

        """
        R = inputs[self.atomic_position_key]  # shape: (n,3)
        idx_i = inputs['idx_i']  # shape: (n_pairs)
        idx_j = inputs['idx_j']  # shape: (n_pairs)
        pair_mask = inputs['pair_mask']  # shape: (n_pairs)

        # Calculate geometric quantities
        r_ij = safe_scale(jax.vmap(lambda i, j: R[i] - R[j])(idx_i, idx_j), scale=pair_mask[:, None])
        # shape: (n_pairs,3)
        if self.pbc:
            lat = inputs[self.lattice_vector_key]  # shape: (3,3)
            r_ij = pbc_diff(r_ij=r_ij, lat=lat)  # shape: (n_pairs,3)

        d_ij = safe_scale(jnp.linalg.norm(r_ij, axis=-1), scale=pair_mask)  # shape : (n_pairs)

        rbf_ij = safe_scale(self.rbf_fn(d_ij[:, None]), scale=pair_mask[:, None])  # shape: (n_pairs,K)
        phi_r_cut = safe_scale(self.cut_fn(d_ij), scale=pair_mask)  # shape: (n_pairs)
        rbf_ij = safe_scale(rbf_ij, scale=phi_r_cut[:, None])  # shape: (n_pairs,K)

        unit_r_ij = safe_mask(mask=d_ij[:, None] != 0,
                              operand=r_ij,
                              fn=lambda y: y / d_ij[:, None],
                              placeholder=0
                              )  # shape: (n_pairs, 3)
        sph_harms_ij = []
        for sph_fn in self.sph_fns:
            sph_ij = safe_scale(sph_fn(unit_r_ij), scale=pair_mask[:, None])  # shape: (n_pairs,2l+1)
            sph_harms_ij += [sph_ij]  # len: |L| / shape: (n_pairs,2l+1)

        sph_harms_ij = jnp.concatenate(sph_harms_ij, axis=-1) if len(self.degrees) > 0 else None
        # shape: (n_pairs,m_tot)

        geometric_data = {'R': R,
                          'r_ij': r_ij,
                          'unit_r_ij': unit_r_ij,
                          'd_ij': d_ij,
                          'rbf_ij': rbf_ij,
                          'phi_r_cut': phi_r_cut,
                          'sph_ij': sph_harms_ij,
                          }
        if self.sphc:
            z = inputs[self.atomic_type_key]
            point_mask = inputs['point_mask']
            geometric_data.update(_init_sphc(z=z,
                                             sph_ij=sph_harms_ij,
                                             phi_r_cut=phi_r_cut,
                                             idx_i=idx_i,
                                             point_mask=point_mask)
                                  )

        if self.solid_harmonic:
            g_ij = sph_harms_ij[:, :, None] * rbf_ij[:, None, :]  # shape: (n_pair,m_tot,K)
            g_ij = safe_scale(g_ij, scale=pair_mask[:, None, None], placeholder=0)  # shape: (n_pair,m_tot,K)
            geometric_data.update({'g_ij': g_ij})

        return geometric_data

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {'degrees': self.degrees,
                                   'radial_basis_function': self.radial_basis_function,
                                   'n_rbf': self.n_rbf,
                                   'radial_cutoff_fn': self.radial_cutoff_fn,
                                   'r_cut': self.r_cut,
                                   'sphc': self.sphc,
                                   'pbc': self.pbc,
                                   'solid_harmonic': self.solid_harmonic,
                                   'prop_keys': self.prop_keys}
                }


class AtomCenteredBasisFunctionEmbed(BaseSubModule):
    prop_keys: Dict
    D: int
    I_min: float
    I_max: float
    # abf: str
    n_abf: int
    num_embeddings: int = 100
    coefficient_std: float = .025
    exponent_std: float = .025
    module_name: str = 'atom_centered_basis_function_embed'

    def setup(self) -> None:
        self.atomic_type_key = self.prop_keys['atomic_type']
        self.d_k = jnp.linspace(self.I_min, self.I_max, self.D)  # shape: (D)
        # self.abf_fn = get_basis_fn(self.abf)
        self.delta = jnp.abs(self.d_k[0] - self.d_k[1])
        self.init_coefficient_fn = nn.initializers.normal(stddev=self.coefficient_std)
        self.init_exponent_fn = nn.initializers.normal(stddev=self.exponent_std)

    @nn.compact
    def __call__(self, inputs, *args, **kwargs):
        z = inputs[self.atomic_type_key]  # shape: (n)
        d_ij = inputs['d_ij']  # shape: (n_pairs)
        idx_i = inputs['idx_i']  # shape: (n_pairs)
        idx_j = inputs['idx_j']  # shape: (n_pairs)

        c = nn.Embed(num_embeddings=self.num_embeddings,
                     features=self.n_abf,
                     embedding_init=self.init_coefficient_fn)(z)  # shape: (n,n_abf)

        c_i = c[idx_i][:, None, :]  # shape: (n_pairs,1,n_abf)
        c_j = c[idx_j][:, None, :]  # shape: (n_pairs,1,n_abf)

        e = nn.Embed(num_embeddings=self.num_embeddings,
                     features=self.n_abf,
                     embedding_init=self.init_exponent_fn)(z)  # shape: (n,n_abf)
        e_j = e[idx_j][:, None, :]  # shape: (n_pairs,1,n_abf)
        e_i = e[idx_i][:, None, :]  # shape: (n_pairs,1,n_abf)

        offsets = ((d_ij[:, None] - self.d_k) ** 2)[..., None]  # shape: (n_pairs,D,1)
        phi_d_ij = jnp.abs(c_j) * jnp.exp(-jnp.abs(e_j) * offsets)  # shape: (n_pairs,D,n_abf)
        phi_0_ij = jnp.abs(c_i) * jnp.exp(-jnp.abs(e_i) * (0. - self.d_k)[:, None])  # shape: (n_pairs,D,n_abf)

        overlap_ij = self.delta * (phi_d_ij * phi_0_ij).sum(axis=-1)  # shape: (n_pairs,D)
        return {'overlap_ij': overlap_ij}

    def __dict_repr__(self):
        return {self.module_name: {'D': self.D,
                                   'I_min': self.I_min,
                                   'I_max': self.I_max,
                                   'n_abf': self.n_abf,
                                   'num_embeddings': self.num_embeddings,
                                   'coefficient_std': self.coefficient_std,
                                   'exponent_std': self.exponent_std,
                                   'prop_keys': self.prop_keys}
                }


class MinimalImageConvention(BaseSubModule):
    prop_keys: Dict
    module_name: str = 'minimal_image_convention'

    def setup(self) -> None:
        self.lattice_vector_key = self.prop_keys['lattice_vector']

    @nn.compact
    def __call__(self, inputs, *args, **kwargs):
        r_ij = inputs['r_ij']  # shape: (n_pairs,3)
        lattice_vectors = inputs[self.lattice_vector_key]  # shape: (3,3)
        r_ij_ = pbc_diff(r_ij=r_ij, lat=lattice_vectors)  # shape: (n_pairs,3)
        return {'r_ij': r_ij_}

    def __dict_repr__(self):
        return {self.module_name: {'prop_keys': self.prop_keys}
                }


class VectorFeatureEmbed(BaseSubModule):
    features: int
    prop_keys: Dict
    module_name: str = 'vector_feature_embed'

    def setup(self) -> None:
        self.atomic_type_key = self.prop_keys['atomic_type']

    def __call__(self, inputs, *args, **kwargs):
        z = inputs[self.atomic_type_key]
        return {'v': jnp.zeros((z.shape[-1], 3, self.features))}

    def __dict_repr__(self):
        return {self.module_name: {'features': self.features,
                                   'prop_keys': self.prop_keys}
                }


def _init_sphc(z, sph_ij, phi_r_cut, idx_i, point_mask, *args, **kwargs):
    _sph_harms_ij = safe_scale(sph_ij, phi_r_cut[:, None])  # shape: (n_pairs,m_tot)
    c = segment_sum(phi_r_cut, segment_ids=idx_i, num_segments=len(z))[:, None]  # shape: (n,1)
    chi = safe_mask(mask=c != 0,
                    operand=segment_sum(_sph_harms_ij, segment_ids=idx_i, num_segments=len(z)),
                    fn=lambda y: y / c,
                    placeholder=0
                    )  # shape: (n,m_tot)

    chi = safe_scale(chi, scale=point_mask[:, None])  # shape: (n,m_tot)
    return {'chi': chi}


class AtomTypeEmbed(BaseSubModule):
    num_embeddings: int
    features: int
    prop_keys: Dict
    module_name: str = 'atom_type_embed'

    def setup(self):
        self.atomic_type_key = self.prop_keys.get('atomic_type')

    @nn.compact
    def __call__(self, inputs: Dict, *args, **kwargs) -> jnp.ndarray:
        """
        Create atomic embeddings based on the atomic types.

        Args:
            inputs (Dict):
                z (Array): atomic types, shape: (n)
                point_mask (Array): Mask for atom-wise operations, shape: (n)
            *args (Tuple):
            **kwargs (Dict):

        Returns: Atomic embeddings, shape: (n,F)

        """
        z = inputs[self.atomic_type_key]
        point_mask = inputs['point_mask']

        z = z.astype(jnp.int32)  # shape: (n)
        return safe_scale(nn.Embed(num_embeddings=self.num_embeddings, features=self.features)(z),
                          scale=point_mask[:, None])

    def __dict_repr__(self):
        return {self.module_name: {'num_embeddings': self.num_embeddings,
                                   'features': self.features,
                                   'prop_keys': self.prop_keys}}


class ChargeSpinEmbed(nn.Module):
    num_embeddings: int
    features: int

    @nn.compact
    def __call__(self,
                 z: jnp.ndarray,
                 psi: jnp.ndarray,
                 point_mask: jnp.ndarray,
                 *args,
                 **kwargs) -> jnp.ndarray:
        """
        Create atomic embeddings based on the total charge or the number of unpaired spins in the system, following the
        embedding procedure introduced in SpookyNet. Returns per atom embeddings of dimension F.

        Args:
            z (Array): Atomic types, shape: (n)
            psi (Array): Total charge or number of unpaired spins, shape: (1)
            point_mask (Array): Mask for atom-wise operations, shape: (n)
            *args ():
            **kwargs ():

        Returns: Per atom embedding, shape: (n,F)

        """
        z = z.astype(jnp.int32)  # shape: (n)
        q = nn.Embed(num_embeddings=self.num_embeddings, features=self.features)(z)  # shape: (n,F)
        psi_ = psi // jnp.inf  # -1 if psi < 0 and 0 otherwise
        psi_ = psi_.astype(jnp.int32)  # shape: (1)
        k = nn.Embed(num_embeddings=2, features=self.features)(psi_)  # shape: (1,F)
        v = nn.Embed(num_embeddings=2, features=self.features)(psi_)  # shape: (1,F)
        q_x_k = (q*k).sum(axis=-1) / jnp.sqrt(self.features)  # shape: (n)
        q_x_k = safe_scale(q_x_k,
                           scale=point_mask,
                           placeholder=-1e10)  # shape: (n)

        numerator = jnp.log(1 + jnp.exp(q_x_k))  # shape: (n)
        a = psi * numerator / numerator.sum(axis=-1)  # shape: (n)
        e_psi = MLP(features=[self.features, self.features],
                    activation_fn=silu,
                    use_bias=False)(a[:, None] * v)  # shape: (n,F)

        return safe_scale(e_psi, scale=point_mask[:, None])  # shape: (n,F)


class ChargeEmbed(BaseSubModule):
    num_embeddings: int
    features: int
    prop_keys: Dict
    module_name: str = 'charge_embed'

    def setup(self):
        self.total_charge_key = self.prop_keys.get('total_charge')
        self.atomic_type_key = self.prop_keys.get('atomic_type')

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs):
        """

        Args:
           inputs (Dict):
                z (Array): atomic types, shape: (n)
                Q (Array): total charge, shape: (1)
                point_mask (Array): Mask for atom-wise operations, shape: (n)
            *args ():
            **kwargs ():

        Returns:

        """
        z = inputs[self.atomic_type_key]
        Q = inputs[self.total_charge_key]
        point_mask = inputs['point_mask']

        return ChargeSpinEmbed(num_embeddings=self.num_embeddings,
                               features=self.features)(z=z, psi=Q, point_mask=point_mask)

    def __dict_repr__(self):
        return {self.module_name: {'num_embeddings': self.num_embeddings,
                                   'features': self.features,
                                   'prop_keys': self.prop_keys}}


class SpinEmbed(BaseSubModule):
    num_embeddings: int
    features: int
    prop_keys: Dict
    module_name: str = 'spin_embed'

    def setup(self):
        self.spin_key = self.prop_keys.get('total_spin')
        self.atomic_type_key = self.prop_keys.get('atomic_type')

    @nn.compact
    def __call__(self,
                 inputs: Dict,
                 *args,
                 **kwargs):
        """
        Args:

            inputs (Dict):
                z (Array): atomic types, shape: (n)
                S (Array): number of unpaired spins, shape: (1)
                point_mask (Array): Mask for atom-wise operations, shape: (n)
            *args ():
            **kwargs ():

        Returns:

        """
        z = inputs[self.atomic_type_key]
        S = inputs[self.spin_key]
        point_mask = inputs['point_mask']

        return ChargeSpinEmbed(num_embeddings=self.num_embeddings,
                               features=self.features)(z=z, psi=S, point_mask=point_mask)

    def __dict_repr__(self):
        return {self.module_name: {'num_embeddings': self.num_embeddings,
                                   'features': self.features,
                                   'prop_keys': self.prop_keys}}
