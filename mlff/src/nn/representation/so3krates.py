import logging
from typing import (Dict, Sequence)
from mlff.src.nn.stacknet import StackNet
from mlff.src.nn.layer.so3krates_layer import So3kratesLayer
from mlff.src.nn.observable import Energy
from mlff.src.nn.embed import AtomTypeEmbed, GeometryEmbed

logging.basicConfig(level=logging.INFO)


def init_so3krates(prop_keys: Dict[str, str],
                   r_cut: float,
                   per_atom_scale: Sequence[float],
                   per_atom_shift: Sequence[float],
                   path_to_cg: str):
    F = 123
    degrees = [1, 2, 3]
    n_layer = 6

    embeddings = [AtomTypeEmbed(num_embeddings=100, features=F, prop_keys=prop_keys)]
    geometry_embedding = GeometryEmbed(degrees=degrees,
                                       radial_basis_function='phys',
                                       n_rbf=32,
                                       radial_cutoff_fn='cosine_cutoff_fn',
                                       r_cut=r_cut,
                                       prop_keys=prop_keys
                                       )
    so3krates_layer = [So3kratesLayer(fb_filter='radial_spherical',
                                      fb_rad_filter_features=[128, F],
                                      fb_sph_filter_features=[32, F],
                                      fb_attention='conv_att',
                                      gb_filter='radial_spherical',
                                      gb_rad_filter_features=[128, F],
                                      gb_sph_filter_features=[32, F],
                                      gb_attention='conv_att',
                                      degrees=degrees,
                                      n_heads=4,
                                      chi_cut=None,
                                      cg_path=path_to_cg
                                      ) for _ in range(n_layer)]
    obs = [Energy(per_atom_scale=per_atom_scale, per_atom_shift=per_atom_shift, prop_keys=prop_keys)]
    net = StackNet(geometry_embedding=geometry_embedding,
                   feature_embeddings=embeddings,
                   layers=so3krates_layer,
                   observables=obs)
    return net
