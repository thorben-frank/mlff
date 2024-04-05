import jax.numpy as jnp
import flax.linen as nn
import jax

import e3x
from functools import partial
from typing import Optional, Sequence

from mlff.nn.base.sub_module import BaseSubModule
from mlff.nn.layer.utils import Residual


def get_activation_fn(name: str):
    return getattr(e3x.nn, name) if name != 'identity' else lambda u: u


def aggregate_from_features(features: Sequence, aggregation: str):
    if aggregation == 'last':
        return features[-1]
    elif aggregation == 'concatenation':
        return jnp.concatenate(features, axis=-1)
    else:
        raise ValueError(f'{aggregation} not a valid aggregation for features.')


def aggregation_from_connectivity(connectivity: str):
    if connectivity == 'independent':
        return 'last'
    elif connectivity == 'skip':
        return 'last'
    elif connectivity == 'dense':
        return 'concatenation'
    else:
        ValueError(f'f{connectivity} not a valid connectivity pattern.')


class ITPLayer(BaseSubModule):
    """Message passing sweep, followed by multiple atom-wise iterated tensor products.

    """

    filter_num_layers: int = 1
    filter_activation_fn: str = 'identity'

    mp_max_degree: Optional[int] = None
    mp_post_res_block: bool = False
    mp_post_res_block_activation_fn: str = 'identity'

    itp_max_degree: Optional[int] = None
    itp_num_features: Optional[int] = None
    itp_num_updates: int = 1
    itp_post_res_block: bool = False
    itp_post_res_block_activation_fn: str = 'identity'
    itp_connectivity: str = 'skip'  # dense, independent
    itp_growth_rate: Optional[int] = None
    itp_dense_final_concatenation: bool = False

    message_normalization: Optional[str] = None  # avg_num_neighbors
    avg_num_neighbors: Optional[float] = None

    feature_collection_over_layers: str = 'last'  # summation

    include_pseudotensors: bool = False
    module_name: str = 'itp_layer'

    def setup(self):
        if self.itp_connectivity == 'dense':
            if self.itp_max_degree is not None:
                if self.itp_max_degree != self.mp_max_degree:
                    raise ValueError(
                        f'For {self.itp_connectivity=} maximal degree of tensor products must be equal '
                        f'to maximal degree in message passing, but {self.itp_max_degree=} != {self.mp_max_degree=}.'
                    )
        if self.message_normalization == 'avg_num_neighbors':
            if self.avg_num_neighbors is None:
                raise ValueError(
                    f'For {self.message_normalization=} average number of neighbors is required, but it is'
                    f'{self.avg_num_neighbors}.'
                )
        if self.itp_growth_rate is not None:
            if self.itp_connectivity != 'dense':
                raise ValueError(
                    f'For {self.itp_connectivity =} the growth_rate can not be set. Use self.itp_connectivity = dense '
                    f'to use growth rate.'
                )

        if self.itp_dense_final_concatenation is True:
            if self.itp_connectivity != 'dense':
                raise ValueError(
                    f'For {self.itp_dense_final_concatenation =} self.itp_connectivity must be set to `dense`.'
                )

    @nn.compact
    def __call__(self,
                 x: jnp.ndarray,
                 basis: jnp.ndarray,
                 cut: jnp.ndarray,
                 idx_i: jnp.ndarray,
                 idx_j: jnp.ndarray,
                 *args,
                 **kwargs):
        """

        Args:
            x (Array): (N, num_features) in first layer
            basis ():
            cut ():
            idx_i ():
            idx_j ():
            *args ():
            **kwargs ():

        Returns:

        """
        num_features = x.shape[-1]
        features = []
        # In the first layer x has shape (N, num_features)
        if x.ndim == 2:
            x = x[:, None, None, :]  # (N, 1, 1, num_features)

        # One layer is applied in the e3x.MessagePass to align basis and feature dimension.
        for _ in range(self.filter_num_layers - 1):
            sigma = get_activation_fn(self.filter_activation_fn)
            basis = sigma(
                e3x.nn.Dense(
                    features=num_features,
                )(
                    basis,
                )
            )

        y = e3x.nn.MessagePass(
            include_pseudotensors=self.include_pseudotensors,
            max_degree=self.mp_max_degree
        )(
            inputs=x,
            basis=basis,
            dst_idx=idx_i,
            src_idx=idx_j,
            num_segments=len(x)
        )

        if self.message_normalization == 'avg_num_neighbors':
            y = jnp.divide(y,
                           jnp.sqrt(jnp.asarray(self.avg_num_neighbors, dtype=y.dtype))
                           )

        # Dense layer to outer part of skip connection.
        z = e3x.nn.Dense(features=num_features)(x)  # (N, 1 or 2, (max_degree + 1)^2, num_features)

        # Skip connection around message pass.
        y = e3x.nn.add(y, z)  # (N, 1 or 2, (max_degree + 1)^2, num_features)

        # Residual block.
        if self.mp_post_res_block:
            y = Residual(
                activation_fn=get_activation_fn(self.mp_post_res_block_activation_fn)
            )(y)

        if self.itp_num_features is not None:
            y = e3x.nn.Dense(
                features=self.itp_num_features
            )(y)

        features.append(y)

        aggregation = aggregation_from_connectivity(self.itp_connectivity)
        for i in range(self.itp_num_updates):
            x_pre_itp = partial(aggregate_from_features, aggregation=aggregation)(features)

            x_itp = e3x.nn.TensorDense(
                include_pseudotensors=False if i == self.itp_num_updates - 1 else self.include_pseudotensors,
                max_degree=0 if i == self.itp_num_updates - 1 else self.itp_max_degree,
                features=self.itp_growth_rate
            )(x_pre_itp)

            if self.itp_post_res_block:
                x_itp = Residual(
                    activation_fn=get_activation_fn(self.itp_post_res_block_activation_fn)
                )(x_itp)

            if self.itp_connectivity == 'skip':
                if i == (self.itp_num_updates - 1):
                    x_pre_itp = e3x.nn.change_max_degree_or_type(
                        x_pre_itp,
                        max_degree=0,
                        include_pseudotensors=False
                    )

                x_itp = e3x.nn.add(x_pre_itp, x_itp)

            # Nasty but necessary for backward compatibility for now.
            if self.itp_dense_final_concatenation:
                assert self.itp_connectivity == 'dense'

                if i == (self.itp_num_updates - 1):
                    x_pre_itp = e3x.nn.change_max_degree_or_type(
                        x_pre_itp,
                        include_pseudotensors=False,
                        max_degree=0
                    )
                    # In the final layer, the output of the ITP is max_degree = 0. Thus, we make the pre-ITP embeddings
                    # also max_degree = 0. The pre-ITP embeddings are a concatenation of all features up to the last
                    # ITP, so it has dimension itp_num_features + (num_itp_updates - 1) * growth_rate. The final output
                    x_itp = partial(aggregate_from_features, aggregation=aggregation)(
                        [x_pre_itp, x_itp]
                    )

            features.append(x_itp)

        if self.feature_collection_over_layers == 'last':
            x_final = features[-1]

            # If num_itp_updates == 0, we have to remove the l>0 degrees from the MP update explicitly.
            x_final = e3x.nn.change_max_degree_or_type(
                x_final,
                max_degree=0,
                include_pseudotensors=False
            )
            # (N, 1, 1, num_itp_features * num_itp_updates) for dense
            # (N, 1, 1, num_itp_updates) for skip and independent

        elif self.feature_collection_over_layers == 'summation':
            x_final = e3x.nn.Dense(
                num_features,
                use_bias=False
            )(x)  # Input is not part of features so add it by hand.
            for x_f in features:
                x_final = e3x.nn.add(
                    x_final,  # First summand.
                    e3x.nn.Dense(
                        num_features,
                        use_bias=False
                    )(
                        e3x.nn.change_max_degree_or_type(
                            x_f,
                            max_degree=0,
                            include_pseudotensors=False
                        )
                    )  # Second summand.
                )  # (N, 1, 1, num_features)
        else:
            raise ValueError(
                f'{self.feature_collection_over_layers} not a valid argument for `feature_collection_over_layers`.'
            )

        x_final = x_final.squeeze(1).squeeze(1)  # (N, num_features)

        return dict(
            x=x_final
        )
