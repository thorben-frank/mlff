from ase.units import *
import jax
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
import portpicker
from typing import Sequence
import wandb
import yaml


def make_so3krates_sparse_from_config(config: config_dict.ConfigDict = None):
    if config is None:
        return nn.SO3kratesSparse()
    else:
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
            qk_non_linearity=model_config.qk_non_linearity,
            activation_fn=model_config.activation_fn,
            layers_behave_like_identity_fn_at_init=model_config.layers_behave_like_identity_fn_at_init,
            output_is_zero_at_init=model_config.output_is_zero_at_init,
            input_convention=model_config.input_convention
        )


def make_optimizer_from_config(config: config_dict.ConfigDict = None):
    if config is None:
        return training_utils.make_optimizer()
    else:
        opt_config = config.optimizer
        return training_utils.make_optimizer(
            name=opt_config.name,
            learning_rate_schedule=opt_config.learning_rate_schedule,
            learning_rate=opt_config.learning_rate,
            learning_rate_schedule_args=opt_config.learning_rate_schedule_args,
            num_of_nans_to_ignore=opt_config.num_of_nans_to_ignore
        )


def run_training(config: config_dict.ConfigDict):
    port = portpicker.pick_unused_port()
    jax.distributed.initialize(f'localhost:{port}', num_processes=1, process_id=0)

    data_filepath = config.data.filepath
    data_filepath = Path(data_filepath).expanduser().absolute().resolve()

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

    if num_train + num_valid > num_data:
        raise ValueError(f"num_train + num_valid = {num_train + num_valid} exceeds the number of data points {num_data}"
                         f" in {data_filepath}.")

    if config.data.energy_shifts is None:
        config.data.energy_shifts = config_dict.placeholder(dict)
        energy_mean = data.transformations.calculate_energy_mean(all_data[:num_train]) * energy_unit
        num_nodes = data.transformations.calculate_average_number_of_nodes(all_data[:num_train])
        energy_shifts = {a: float(energy_mean / num_nodes) for a in range(119)}
        config.data.energy_shifts = energy_shifts

    training_data = data.transformations.subtract_atomic_energy_shifts(
        data.transformations.unit_conversion(
            all_data[:num_train],
            energy_unit=energy_unit,
            length_unit=length_unit
        ),
        atomic_energy_shifts=config.data.energy_shifts
    )

    validation_data = data.transformations.subtract_atomic_energy_shifts(
        data.transformations.unit_conversion(
            all_data[num_train:(num_train + num_valid)],
            energy_unit=energy_unit,
            length_unit=length_unit
        ),
        atomic_energy_shifts=config.data.energy_shifts
    )

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

        batch_max_num_nodes = data_stats['max_num_of_nodes'] * config.training.batch_max_num_graphs
        batch_max_num_edges = data_stats['max_num_of_edges'] * config.training.batch_max_num_graphs

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
    )


def run_evaluation(config, num_test: int = None, testing_targets: Sequence[str] = None):
    port = portpicker.pick_unused_port()
    jax.distributed.initialize(f'localhost:{port}', num_processes=1, process_id=0)

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

        batch_max_num_nodes = data_stats['max_num_of_nodes'] * config.training.batch_max_num_graphs
        batch_max_num_edges = data_stats['max_num_of_edges'] * config.training.batch_max_num_graphs

        config.training.batch_max_num_nodes = batch_max_num_nodes
        config.training.batch_max_num_edges = batch_max_num_edges

    ckpt_dir = Path(config.workdir) / 'checkpoints'
    ckpt_dir = ckpt_dir.expanduser().absolute().resolve()
    ckpt_mngr = checkpoint.CheckpointManager(
        ckpt_dir,
        {'params': checkpoint.AsyncCheckpointer(checkpoint.PyTreeCheckpointHandler())},
        options=checkpoint.CheckpointManagerOptions(step_prefix='ckpt')
    )
    latest_step = ckpt_mngr.latest_step()
    if latest_step is not None:
        params = ckpt_mngr.restore(
            latest_step,
            {'params': checkpoint.AsyncCheckpointer(checkpoint.PyTreeCheckpointHandler())}
        )['params']
    else:
        raise FileNotFoundError(f'No checkpoint found at {ckpt_dir}.')

    so3k = make_so3krates_sparse_from_config(config)
    print(f'Evaluate on {data_filepath} for targets {targets}.')
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
