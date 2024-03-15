import flax.linen as nn
import jax
from mlff.nn.stacknet import StackNetSparse
from mlff.nn.embed import GeometryEmbedE3x, AtomTypeEmbedSparse
from mlff.nn.layer import ITPLayer
from mlff.nn.observable import EnergySparse
from typing import Optional, Sequence


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
        message_normalization: Optional[str] = None,
        avg_num_neighbors: Optional[float] = None,
        feature_collection_over_layers: str = 'final',
        include_pseudotensors: bool = False,
        output_is_zero_at_init: bool = True,
        energy_regression_dim: int = 128,
        energy_activation_fn: str = 'identity',
        energy_learn_atomic_type_scales: bool = False,
        energy_learn_atomic_type_shifts: bool = False,
        input_convention: str = 'positions'
):
    atom_type_embed = AtomTypeEmbedSparse(
        num_features=num_features,
        prop_keys=None
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
        message_normalization=message_normalization,
        avg_num_neighbors=avg_num_neighbors,
        feature_collection_over_layers=feature_collection_over_layers,
        include_pseudotensors=include_pseudotensors
    )]

    energy = EnergySparse(
        prop_keys=None,
        output_is_zero_at_init=output_is_zero_at_init,
        regression_dim=energy_regression_dim,
        activation_fn=getattr(
            nn.activation, energy_activation_fn
        ) if energy_activation_fn != 'identity' else lambda u: u,
        learn_atomic_type_scales=energy_learn_atomic_type_scales,
        learn_atomic_type_shifts=energy_learn_atomic_type_shifts,
    )

    return StackNetSparse(
        geometry_embeddings=[geometry_embed],
        feature_embeddings=[atom_type_embed],
        layers=layers,
        observables=[energy],
        prop_keys=None
    )
