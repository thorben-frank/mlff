from ase.units import *
import json
from mlff import nn
from mlff import training_utils
from mlff import evaluation_utils
from mlff import data
from mlff import jraph_utils
from mlff.nn.stacknet.observable_function_sparse import get_energy_and_force_fn_sparse
from ml_collections import config_dict
import numpy as np
from orbax import checkpoint
from pathlib import Path
from typing import Sequence
import wandb
import yaml
import logging
from functools import partial, partialmethod

logging.MLFF = 35
logging.addLevelName(logging.MLFF, 'MLFF')
logging.Logger.trace = partialmethod(logging.Logger.log, logging.MLFF)
logging.mlff = partial(logging.log, logging.MLFF)


def make_so3krates_sparse_from_config(config: config_dict.ConfigDict = None):
    """Make a SO3krates model from a config.

    Args:
        config (): The config.

    Returns:
        SO3krates flax model.
    """
    model_config = config.model

    return nn.SO3kratesSparse(
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
        message_normalization=config.model.message_normalization,
        avg_num_neighbors=config.data.avg_num_neighbors if config.model.message_normalization == 'avg_num_neighbors' else None,
        qk_non_linearity=model_config.qk_non_linearity,
        activation_fn=model_config.activation_fn,
        layers_behave_like_identity_fn_at_init=model_config.layers_behave_like_identity_fn_at_init,
        output_is_zero_at_init=model_config.output_is_zero_at_init,
        input_convention=model_config.input_convention,
        energy_regression_dim=model_config.energy_regression_dim,
        energy_activation_fn=model_config.energy_activation_fn,
        energy_learn_atomic_type_scales=model_config.energy_learn_atomic_type_scales,
        energy_learn_atomic_type_shifts=model_config.energy_learn_atomic_type_shifts
    )


def make_optimizer_from_config(config: config_dict.ConfigDict = None):
    """Make optax optimizer from config.

    Args:
        config (): The config.

    Returns:
        optax.Optimizer.
    """
    if config is None:
        return training_utils.make_optimizer()
    else:
        opt_config = config.optimizer
        return training_utils.make_optimizer(
            name=opt_config.name,
            learning_rate_schedule=opt_config.learning_rate_schedule,
            learning_rate=opt_config.learning_rate,
            learning_rate_schedule_args=opt_config.learning_rate_schedule_args if opt_config.learning_rate_schedule_args is not None else dict(),
            num_of_nans_to_ignore=opt_config.num_of_nans_to_ignore,
            gradient_clipping=opt_config.gradient_clipping,
            gradient_clipping_args=opt_config.gradient_clipping_args if opt_config.gradient_clipping_args is not None else dict()
        )


def run_training(config: config_dict.ConfigDict):
    """Run training given a config.

    Args:
        config (): The config.

    Returns:

    """
    data_filepath = config.data.filepath
    data_filepath = Path(data_filepath).expanduser().absolute().resolve()

    if data_filepath.suffix == '.npz':
        loader = data.NpzDataLoaderSparse(input_file=data_filepath)
    elif data_filepath.stem[:5].lower() == 'spice':
        logging.mlff(f'Found SPICE dataset at {data_filepath}.')
        if data_filepath.suffix != '.hdf5':
            raise ValueError(
                f'Loader assumes that SPICE is in hdf5 format. Found {data_filepath.suffix} as'
                f'suffix.')
        loader = data.SpiceDataLoaderSparse(input_file=data_filepath)
    else:
        loader = data.AseDataLoaderSparse(input_file=data_filepath)

    energy_unit = eval(config.data.energy_unit)
    length_unit = eval(config.data.length_unit)

    # Cutoff is in Angstrom, so we have to divide the cutoff by the length unit.
    all_data, data_stats = loader.load_all(cutoff=config.model.cutoff / length_unit)
    num_data = len(all_data)

    split_seed = config.data.split_seed
    numpy_rng = np.random.RandomState(split_seed)
    numpy_rng.shuffle(all_data)

    num_train = config.training.num_train
    num_valid = config.training.num_valid

    if num_train + num_valid > num_data:
        raise ValueError(f"num_train + num_valid = {num_train + num_valid} exceeds the number of data points {num_data}"
                         f" in {data_filepath}.")

    if config.data.shift_mode == 'mean':
        config.data.energy_shifts = config_dict.placeholder(dict)
        energy_mean = data.transformations.calculate_energy_mean(all_data[:num_train]) * energy_unit
        num_nodes = data.transformations.calculate_average_number_of_nodes(all_data[:num_train])
        energy_shifts = {str(a): float(energy_mean / num_nodes) for a in range(119)}
        config.data.energy_shifts = energy_shifts
    elif config.data.shift_mode == 'custom':
        if config.data.energy_shifts is None:
            raise ValueError('For config.data.shift_mode == custom config.data.energy_shifts must be given.')
    else:
        config.data.energy_shifts = {str(a): 0. for a in range(119)}

    training_data = data.transformations.subtract_atomic_energy_shifts(
        data.transformations.unit_conversion(
            all_data[:num_train],
            energy_unit=energy_unit,
            length_unit=length_unit
        ),
        atomic_energy_shifts={int(k): v for (k, v) in config.data.energy_shifts.items()}
    )

    validation_data = data.transformations.subtract_atomic_energy_shifts(
        data.transformations.unit_conversion(
            all_data[num_train:(num_train + num_valid)],
            energy_unit=energy_unit,
            length_unit=length_unit
        ),
        atomic_energy_shifts={int(k): v for (k, v) in config.data.energy_shifts.items()}
    )

    # If messages are normalized by the average number of neighbors, we need to calculate this quantity from the
    # training data.
    if config.model.message_normalization == 'avg_num_neighbors':
        config.data.avg_num_neighbors = config_dict.placeholder(float)
        avg_num_neighbors = data.transformations.calculate_average_number_of_neighbors(all_data[:num_train])
        config.data.avg_num_neighbors = np.array(avg_num_neighbors).item()

    opt = make_optimizer_from_config(config)
    so3k = make_so3krates_sparse_from_config(config)

    loss_fn = training_utils.make_loss_fn(
        get_energy_and_force_fn_sparse(so3k),
        weights=config.training.loss_weights
    )

    workdir = Path(config.workdir).expanduser().absolute().resolve()
    workdir.mkdir(exist_ok=config.training.allow_restart)

    if config.training.batch_max_num_nodes is None:
        assert config.training.batch_max_num_edges is None

        batch_max_num_nodes = data_stats['max_num_of_nodes'] * (config.training.batch_max_num_graphs - 1) + 1
        batch_max_num_edges = data_stats['max_num_of_edges'] * (config.training.batch_max_num_graphs - 1) + 1

        config.training.batch_max_num_nodes = batch_max_num_nodes
        config.training.batch_max_num_edges = batch_max_num_edges

    with open(workdir / 'hyperparameters.json', 'w') as fp:
        # json_config = config.to_dict()
        # energy_shifts = json_config['data']['energy_shifts']
        # energy_shifts = jax.tree_map(lambda x: x.item(), energy_shifts)
        json.dump(config.to_dict(), fp)

    with open(workdir / "hyperparameters.yaml", "w") as yaml_file:
        yaml.dump(config.to_dict(), yaml_file, default_flow_style=False)

    wandb.init(config=config.to_dict(), **config.training.wandb_init_args)
    logging.mlff('Training is starting!')
    training_utils.fit(
        model=so3k,
        optimizer=opt,
        loss_fn=loss_fn,
        graph_to_batch_fn=jraph_utils.graph_to_batch_fn,
        batch_max_num_edges=config.training.batch_max_num_edges,
        batch_max_num_nodes=config.training.batch_max_num_nodes,
        batch_max_num_graphs=config.training.batch_max_num_graphs,
        training_data=list(training_data),
        validation_data=list(validation_data),
        ckpt_dir=workdir / 'checkpoints',
        eval_every_num_steps=config.training.eval_every_num_steps,
        allow_restart=config.training.allow_restart,
        num_epochs=config.training.num_epochs,
        training_seed=config.training.training_seed,
        model_seed=config.training.model_seed,
        log_gradient_values=config.training.log_gradient_values
    )
    logging.mlff('Training has finished!')


def run_evaluation(config, num_test: int = None, testing_targets: Sequence[str] = None):
    """Run evaluation, given the config and additional args.

    Args:
        config (): The config file.
        num_test (): Number of testing points. If not given, is determined from config using
            num_test = num_data - num_train - num_valid.
        testing_targets (): Targets used for computing metrics. Defaults to the ones found in
            config.training.loss_weights.

    Returns:
        The metrics on `testing_targets`.
    """
    data_filepath = config.data.filepath
    data_filepath = Path(data_filepath).expanduser().absolute().resolve()

    targets = testing_targets if testing_targets is not None else list(config.training.loss_weights.keys())

    if data_filepath.suffix == '.npz':
        loader = data.NpzDataLoaderSparse(input_file=data_filepath)
    else:
        loader = data.AseDataLoaderSparse(input_file=data_filepath)

    all_data, data_stats = loader.load_all(cutoff=config.model.cutoff)
    num_data = len(all_data)

    energy_unit = eval(config.data.energy_unit)
    length_unit = eval(config.data.length_unit)

    split_seed = config.data.split_seed
    numpy_rng = np.random.RandomState(split_seed)
    numpy_rng.shuffle(all_data)

    num_train = config.training.num_train
    num_valid = config.training.num_valid

    if (num_train + num_valid + num_test) > num_data:
        raise ValueError(
            f'num_train + num_valid + num_test = {num_train + num_valid + num_test} > num_data = {num_data}.'
        )

    upper_bound = (num_train + num_valid + num_test) if num_test is not None else num_data

    testing_data = data.transformations.subtract_atomic_energy_shifts(
        data.transformations.unit_conversion(
            all_data[(num_train + num_valid):upper_bound],
            energy_unit=energy_unit,
            length_unit=length_unit
        ),
        atomic_energy_shifts={int(k): v for (k, v) in config.data.energy_shifts.items()}
    )

    if config.training.batch_max_num_nodes is None:
        assert config.training.batch_max_num_edges is None

        batch_max_num_nodes = data_stats['max_num_of_nodes'] * (config.training.batch_max_num_graphs - 1) + 1
        batch_max_num_edges = data_stats['max_num_of_edges'] * (config.training.batch_max_num_graphs - 1) + 1

        config.training.batch_max_num_nodes = batch_max_num_nodes
        config.training.batch_max_num_edges = batch_max_num_edges

    ckpt_dir = Path(config.workdir) / 'checkpoints'
    ckpt_dir = ckpt_dir.expanduser().absolute().resolve()
    logging.mlff(f'Restore parameters from {ckpt_dir} ...')
    ckpt_mngr = checkpoint.CheckpointManager(
        ckpt_dir,
        {'params': checkpoint.PyTreeCheckpointer()},
        options=checkpoint.CheckpointManagerOptions(step_prefix='ckpt')
    )
    latest_step = ckpt_mngr.latest_step()
    if latest_step is not None:
        params = ckpt_mngr.restore(
            latest_step,
            items=None
        )['params']
    else:
        raise FileNotFoundError(f'No checkpoint found at {ckpt_dir}.')
    logging.mlff(f'... done.')

    so3k = make_so3krates_sparse_from_config(config)
    logging.mlff(f'Evaluate on {data_filepath} for targets {targets}.')
    return evaluation_utils.evaluate(
        model=so3k,
        params=params,
        graph_to_batch_fn=jraph_utils.graph_to_batch_fn,
        testing_data=testing_data,
        testing_targets=targets,
        batch_max_num_nodes=config.training.batch_max_num_nodes,
        batch_max_num_edges=config.training.batch_max_num_edges,
        batch_max_num_graphs=config.training.batch_max_num_graphs,
    )
