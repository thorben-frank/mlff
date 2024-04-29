import argparse
import json
from mlff.config import from_config
from ml_collections import config_dict
import pathlib


def evaluate_itp_net_on():
    # Create the parser
    parser = argparse.ArgumentParser(
        description='Evaluate a ITPNet model on a different data set than the one it has been trained on.'
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
        help='How many nodes to put in a batch. If not set is determined from max_num_graphs and max_num_nodes in '
             '--datafile.'
    )
    parser.add_argument(
        "--max_num_edges",
        type=int,
        default=None,
        required=False,
        help='How many edges to put in a batch. If not set is determined from max_num_graphs and max_num_edges in '
             '--datafile.'
    )
    parser.add_argument( #TODO: remove --max_num_pairs
        "--max_num_pairs",
        type=int,
        default=None,
        required=False,
        help='How many edges to put in a batch. If not set is determined from max_num_graphs and max_num_edges in '
             '--datafile.'
    )
    parser.add_argument(
        '--num_test',
        type=int,
        required=False,
        default=None,
        help='Number of test points to use. If given, the first num_test points in the dataset are used for evaluation.'
    )
    parser.add_argument('--write_batch_metrics_to',
                        type=str,
                        required=False,
                        default=None,
                        help='Path to csv file where metrics per batch should be written to. If not given, '
                             'batch metrics are not written to a file. Note, that the metrics are written per batch, '
                             'so one-to-one correspondence to the original data set can only be achieved when '
                             '`batch_max_num_nodes = 2` which allows one graph per batch, following the `jraph` logic '
                             'that one graph in used as padding graph.'
                        )
    args = parser.parse_args()

    if args.num_test is not None and args.write_batch_metrics_to is not None:
        raise ValueError(
            f'--num_test={args.num_test} is not `None` such that data is randomly sub-sampled from {args.filepath}. '
            f'At the same time `--write_batch_metrics_to={args.write_batch_metrics_to}` is specified. Due to the '
            f'random subsampling of data, there is no one-to-one correspondence between the lines in the file the '
            f'metrics are written to and the indices of the data point, so we raise an error here for security.'
        )

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
    cfg.training.batch_max_num_pairs = args.max_num_pairs

    if args.write_batch_metrics_to is not None and cfg.training.batch_max_num_graphs > 2:
        raise ValueError(
            f'--write_batch_metrics_to={args.write_batch_metrics_to} is not None and `batch_max_num_graphs != 2.` '
            'Note, that the metrics are written per batch, so one-to-one correspondence to the original data set can '
            'only be achieved when `batch_max_num_nodes = 2` which allows one graph per batch, following the `jraph` '
            'logic that one graph in used as padding graph. Raising error for security here.'
        )

    # Expand and resolve path for writing metrics.
    write_batch_metrics_to = pathlib.Path(
        args.write_batch_metrics_to
    ).expanduser().resolve() if args.write_batch_metrics_to is not None else None

    if write_batch_metrics_to.suffix == '.csv':
        pass
    else:
        write_batch_metrics_to = f'{write_batch_metrics_to}.csv'

    metrics = from_config.run_evaluation(
        config=cfg,
        num_test=args.num_test,
        pick_idx=None,
        write_batch_metrics_to=write_batch_metrics_to,
        model='itp_net'
    )
    print(metrics)


if __name__ == '__main__':
    evaluate_itp_net_on()
