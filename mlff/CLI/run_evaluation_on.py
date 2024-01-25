import argparse
import json
from mlff.config import from_config
from ml_collections import config_dict
import pathlib


def evaluate_so3krates_sparse_on():
    # Create the parser
    parser = argparse.ArgumentParser(
        description='Evaluate a SO3kratesSparse model on a different data set than the one it has been trained on.'
    )
    parser.add_argument(
        '--workdir', type=str, required=True, help='Workdir of the model.'
    )
    parser.add_argument(
        '--filepath', type=str, required=True, help='Path to the data file model should be applied to.'
    )
    parser.add_argument(
        "--length_unit",
        type=str,
        required=False,
        default='Angstrom',
        help='Length unit in the data. Defaults to Angstrom.'
    )
    parser.add_argument(
        "--energy_unit",
        type=str,
        required=False,
        default='eV',
        help='Energy unit in the data. Defaults to eV.'
    )
    parser.add_argument(
        "--max_num_graphs",
        type=int,
        default=11,
        required=False,
        help='How many graphs to put in a batch.'
    )
    parser.add_argument(
        "--max_num_nodes",
        type=int,
        default=None,
        required=False,
        help='How many nodes to put in a batch.'
    )
    parser.add_argument(
        "--max_num_edges",
        type=int,
        default=None,
        required=False,
        help='How many edges to put in a batch.'
    )
    parser.add_argument(
        '--num_test',
        type=int,
        required=False,
        default=None,
        help='Number of test points to use. If given, the first num_test points in the dataset are used for evaluation.'
    )

    args = parser.parse_args()

    workdir = pathlib.Path(args.workdir).expanduser().resolve()

    with open(workdir / 'hyperparameters.json', 'r') as fp:
        x = json.load(fp)
    cfg = config_dict.ConfigDict(x)

    # Overwrite the data information in config dict.
    cfg.data.filepath = args.filepath
    cfg.data.length_unit = args.length_unit
    cfg.data.energy_unit = args.energy_unit

    # Set the batching info.
    cfg.training.batch_max_num_graphs = args.max_num_graphs
    cfg.training.batch_max_num_edges = args.max_num_edges
    cfg.training.batch_max_num_nodes = args.max_num_nodes

    metrics = from_config.run_evaluation(config=cfg, num_test=args.num_test, pick_idx=None)
    print(metrics)


if __name__ == '__main__':
    evaluate_so3krates_sparse_on()
