import flax.linen as nn
from mlff.nn.stacknet import StackNetSparse
from mlff.nn.embed import GeometryEmbedE3x
from mlff.nn.layer import ITPLayer
from mlff.nn.observable import EnergySparse, DipoleVecSparse, HirshfeldSparse, PartialChargesSparse, ElectrostaticEnergySparse, DispersionEnergySparse, ZBLRepulsionSparse

from .representation_utils import make_embedding_modules

from typing import Optional


def init_itp_net(
        num_features: int = 32,
        radial_basis_fn: str = 'reciprocal_bernstein',
        num_radial_basis_fn: int = 16,
        cutoff_fn: str = 'smooth_cutoff',
        cutoff: float = 5.,
        filter_num_layers: int = 1,
        filter_activation_fn: str = 'identity',
        mp_max_degree: int = 2,
        mp_post_res_block: bool = True,
        mp_post_res_block_activation_fn: str = 'identity',
        itp_max_degree: int = 2,
        itp_num_features: int = 32,
        itp_num_updates: int = 3,
        itp_post_res_block: bool = True,
        itp_post_res_block_activation_fn: str = 'identity',
        itp_connectivity: str = 'dense',
        itp_growth_rate: int = Optional[None],
        itp_dense_final_concatenation: bool = False,
        message_normalization: Optional[str] = None,
        avg_num_neighbors: Optional[float] = None,
        feature_collection_over_layers: str = 'final',
        include_pseudotensors: bool = False,
        output_is_zero_at_init: bool = True,
        use_charge_embed: bool = False,
        use_spin_embed: bool = False,
        energy_regression_dim: int = 128,
        energy_activation_fn: str = 'identity',
        energy_learn_atomic_type_scales: bool = False,
        energy_learn_atomic_type_shifts: bool = False,
        input_convention: str = 'positions',
        electrostatic_energy_bool: bool = False,
        electrostatic_energy_scale: float = 1.0,
        dispersion_energy_bool: bool = False,
        dispersion_energy_scale: float = 1.0,
        zbl_repulsion_bool: bool = False,
):
    embedding_modules = make_embedding_modules(
        num_features=num_features,
        use_spin_embed=use_spin_embed,
        use_charge_embed=use_charge_embed
    )

    geometry_embed = GeometryEmbedE3x(
        max_degree=mp_max_degree,
        radial_basis_fn=radial_basis_fn,
        num_radial_basis_fn=num_radial_basis_fn,
        cutoff_fn=cutoff_fn,
        cutoff=cutoff,
        input_convention=input_convention,
        prop_keys=None
    )

    layers = [ITPLayer(
        mp_max_degree=mp_max_degree,
        filter_num_layers=filter_num_layers,
        filter_activation_fn=filter_activation_fn,
        mp_post_res_block=mp_post_res_block,
        mp_post_res_block_activation_fn=mp_post_res_block_activation_fn,
        itp_max_degree=itp_max_degree,
        itp_num_features=itp_num_features,
        itp_num_updates=itp_num_updates,
        itp_post_res_block=itp_post_res_block,
        itp_post_res_block_activation_fn=itp_post_res_block_activation_fn,
        itp_connectivity=itp_connectivity,
        itp_growth_rate=itp_growth_rate,
        itp_dense_final_concatenation=itp_dense_final_concatenation,
        message_normalization=message_normalization,
        avg_num_neighbors=avg_num_neighbors,
        feature_collection_over_layers=feature_collection_over_layers,
        include_pseudotensors=include_pseudotensors
    )]

    partial_charges = PartialChargesSparse(
        prop_keys=None,
        output_is_zero_at_init=output_is_zero_at_init,
        regression_dim=energy_regression_dim,
        activation_fn=getattr(
            nn.activation, energy_activation_fn
        ) if energy_activation_fn != 'identity' else lambda u: u,
    )

    hirshfeld_ratios = HirshfeldSparse(
        prop_keys=None,
        output_is_zero_at_init=output_is_zero_at_init,
        regression_dim=energy_regression_dim,
        activation_fn=getattr(
            nn.activation, energy_activation_fn
        ) if energy_activation_fn != 'identity' else lambda u: u,
    ) 

    dispersion_energy = DispersionEnergySparse(
        prop_keys=None,
        output_is_zero_at_init=output_is_zero_at_init,
        hirshfeld_ratios=hirshfeld_ratios,
        regression_dim=energy_regression_dim,
        activation_fn=getattr(
            nn.activation, energy_activation_fn
        ) if energy_activation_fn != 'identity' else lambda u: u,
        dispersion_energy_scale=dispersion_energy_scale,
    )

    electrostatic_energy = ElectrostaticEnergySparse(
        prop_keys=None,
        output_is_zero_at_init=output_is_zero_at_init,
        regression_dim=energy_regression_dim,
        activation_fn=getattr(
            nn.activation, energy_activation_fn
        ) if energy_activation_fn != 'identity' else lambda u: u,
        partial_charges=partial_charges,
        electrostatic_energy_scale=electrostatic_energy_scale,
    )

    dipole_vec = DipoleVecSparse(
        prop_keys=None,
        output_is_zero_at_init=output_is_zero_at_init,
        regression_dim=energy_regression_dim,
        activation_fn=getattr(
            nn.activation, energy_activation_fn
        ) if energy_activation_fn != 'identity' else lambda u: u,
        partial_charges=partial_charges,
    )
    
    zbl_repulsion = ZBLRepulsionSparse(
        prop_keys=None,
    )

    energy = EnergySparse(
        prop_keys=None,
        output_is_zero_at_init=output_is_zero_at_init,
        regression_dim=energy_regression_dim,
        activation_fn=getattr(
            nn.activation, energy_activation_fn
        ) if energy_activation_fn != 'identity' else lambda u: u,
        learn_atomic_type_scales=energy_learn_atomic_type_scales,
        learn_atomic_type_shifts=energy_learn_atomic_type_shifts,
        electrostatic_energy=electrostatic_energy,
        electrostatic_energy_bool=electrostatic_energy_bool,
        dispersion_energy=dispersion_energy,
        dispersion_energy_bool=dispersion_energy_bool,
        zbl_repulsion=zbl_repulsion,
        zbl_repulsion_bool=zbl_repulsion_bool
    )

    return StackNetSparse(
        geometry_embeddings=[geometry_embed],
        feature_embeddings=embedding_modules,
        layers=layers,
        observables=[energy, dipole_vec, hirshfeld_ratios],
        prop_keys=None
    )
