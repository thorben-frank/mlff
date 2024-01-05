from mlff.nn import SO3kratesSparse
from ml_collections import ConfigDict

import optax


def make_so3krates_sparse_from_config(config: ConfigDict = None):
    if config is None:
        return SO3kratesSparse()
    else:
        model_config = config.model

        return SO3kratesSparse(
            num_layers=model_config.num_layers,
            num_features=model_config.num_features,
            num_heads=model_config.num_heads,
            num_features_head=model_config.num_features_head,
            radial_basis_fn=model_config.radial_basis_fn,
            num_radial_basis_fn=model_config.num_radial_basis_fn,
            cutoff_fn=model_config.cutoff_fn,
            cutoff=model_config.cutoff,
            degrees=model_config.degrees,
            residual_mlp_1=model_config.residual_mlp_1,
            residual_mlp_2=model_config.residual_mlp_2,
            layer_normalization_1=model_config.layer_normalization_1,
            layer_normalization_2=model_config.layer_normalization_2,
            qk_non_linearity=model_config.qk_non_linearity,
            activation_fn=model_config.activation_fn,
            layers_behave_like_identity_fn_at_init=model_config.layers_behave_like_identity_fn_at_init,
            output_is_zero_at_init=model_config.output_is_zero_at_init,
            input_convention=model_config.input_convention
        )


def make_optimizer_from_config(config: ConfigDict):
    opt_config = config.optimizer
    opt = getattr(optax, opt_config.name)

    lr_schedule = getattr(optax, opt_config.learning_rate_schedule)
    lr_schedule = lr_schedule(opt_config.learning_rate, **opt_config.learning_rate_schedule_args)

    opt = opt(lr_schedule)

    return optax.apply_if_finite(opt, max_consecutive_errors=opt_config.num_of_nans_to_ignore)
