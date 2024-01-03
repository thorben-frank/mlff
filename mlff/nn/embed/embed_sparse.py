import jax
import jax.numpy as jnp

from functools import partial
from typing import (Any, Dict, Sequence)

import flax.linen as nn

from mlff.nn.base.sub_module import BaseSubModule
from mlff.masking.mask import safe_mask
from mlff.masking.mask import safe_norm
from mlff.basis_function.radial import get_rbf_fn
from mlff.cutoff_function import add_cell_offsets_sparse
from mlff.cutoff_function import get_cutoff_fn
from mlff.basis_function.spherical import init_sph_fn


class GeometryEmbedSparse(BaseSubModule):
    prop_keys: Dict
    degrees: Sequence[int]
    radial_basis_fn: str
    num_radial_basis_fn: int
    cutoff_fn: str
    cutoff: float
    input_convention: str = 'positions'
    module_name: str = 'geometry_embed_sparse'

    def setup(self):
        self.ylm_fns = [init_sph_fn(y) for y in self.degrees]

        self.rbf_fn = get_rbf_fn(self.radial_basis_fn)(
            n_rbf=self.num_radial_basis_fn,
            r_cut=self.cutoff
        )

        self.cut_fn = partial(get_cutoff_fn(self.cutoff_fn), r_cut=self.cutoff)

    def __call__(self, inputs: Dict):
        """
        Embed geometric information from the atomic positions and its neighboring atoms.

        Args:
            inputs (Dict): Input dictionary, with key - entry pairs:
                positions (Array): Atomic positions, (N, 3)
                idx_i (Array): Index centering atom, (num_pairs)
                idx_j (Array): Index neighboring atom, (num_pairs)
                cell (Array): Unit or super cell, (num_pairs, 3, 3)
                cell_offsets: Cell offsets for PBCs, (num_pairs, 3)

        Returns:
        """
        idx_i = inputs['idx_i']  # shape: (num_pairs)
        idx_j = inputs['idx_j']  # shape: (num_pairs)
        cell = inputs['cell']  # shape: (num_graphs, 3, 3)
        cell_offsets = inputs['cell_offset']  # shape: (num_pairs, 3)

        if self.input_convention == 'positions':
            positions = inputs['positions']  # (N, 3)

            # Calculate pairwise distance vectors
            r_ij = jax.vmap(
                lambda i, j: positions[j] - positions[i]
            )(idx_i, idx_j)  # (num_pairs, 3)

            # Apply minimal image convention if needed.
            if cell is not None:
                r_ij = add_cell_offsets_sparse(
                    r_ij=r_ij,
                    cell=cell,
                    cell_offsets=cell_offsets
                )  # shape: (num_pairs,3)

        elif self.input_convention == 'displacements':
            positions = None
            r_ij = inputs['displacements']
        else:
            raise ValueError(f"{self.input_convention} is not a valid argument for `input_convention`.")

        # Calculate pairwise distances.
        d_ij = safe_norm(r_ij, axis=-1)  # shape : (num_pairs)

        # Gaussian basis expansion of distances.
        rbf_ij = self.rbf_fn(jnp.expand_dims(d_ij, axis=-1))  # shape: (num_pairs, num_radial_basis_fn)

        # Radial cutoff function.
        cut = self.cut_fn(d_ij)  # shape: (num_pairs)

        # Normalized distance vectors.
        unit_r_ij = safe_mask(mask=jnp.expand_dims(d_ij, axis=-1) > 0,
                              operand=r_ij,
                              fn=lambda y: y / jnp.expand_dims(d_ij, axis=-1),
                              placeholder=0
                              )  # shape: (num_pairs, 3)

        # Spherical harmonics.
        ylm_ij = [ylm_fn(unit_r_ij) for ylm_fn in self.ylm_fns]
        ylm_ij = jnp.concatenate(ylm_ij, axis=-1) if len(ylm_ij) > 0 else None  # shape: (num_pairs, m_tot)

        geometric_data = {'positions': positions,
                          'r_ij': r_ij,
                          'unit_r_ij': unit_r_ij,
                          'd_ij': d_ij,
                          'rbf_ij': rbf_ij,
                          'cut': cut,
                          'ylm_ij': ylm_ij,
                          'ev': jnp.zeros((len(positions), ylm_ij.shape[-1]), dtype=positions.dtype)
                          }

        return geometric_data

    def reset_input_convention(self, input_convention: str) -> None:
        self.input_convention = input_convention

    def __dict_repr__(self) -> Dict[str, Dict[str, Any]]:
        return {self.module_name: {'degrees': self.degrees,
                                   'radial_basis_fn': self.radial_basis_fn,
                                   'num_radial_basis_fn': self.num_radial_basis_fn,
                                   'cutoff_fn': self.cutoff_fn,
                                   'cutoff': self.cutoff,
                                   'input_convention': self.input_convention,
                                   'prop_keys': self.prop_keys}
                }


class AtomTypeEmbedSparse(BaseSubModule):
    num_features: int
    prop_keys: Dict
    zmax: int = 118
    module_name: str = 'atom_type_embed_sparse'

    @nn.compact
    def __call__(self, inputs: Dict, *args, **kwargs) -> jnp.ndarray:
        """
        Create atomic embeddings based on the atomic types.

        Args:
            inputs (Dict):
                atomic_numbers (Array): atomic types, shape: (num_nodes)
            *args (Tuple):
            **kwargs (Dict):

        Returns: Atomic embeddings, shape: (num_nodes,num_features)

        """
        atomic_numbers = inputs['atomic_numbers']
        return nn.Embed(num_embeddings=self.zmax + 1, features=self.num_features)(atomic_numbers)

    def __dict_repr__(self):
        return {self.module_name: {'num_features': self.num_features,
                                   'zmax': self.zmax,
                                   'prop_keys': self.prop_keys}}
