import argparse
import json
from mlff.config import from_config
from ml_collections import config_dict
import pathlib


def evaluate_so3krates_sparse():
    # Create the parser
    parser = argparse.ArgumentParser(description='Evaluate a SO3kratesSparse model.')
    parser.add_argument('--workdir', type=str, required=True, help='workdir')
    parser.add_argument('--num_test', type=int, required=False, default=None, help='Number of test points to use.')
    parser.add_argument(
        '--on_split',
        type=str,
        required=False,
        default='test',
        help='On which split to evaluate. Options are (training, validation, test or full).'
    )

    args = parser.parse_args()

    workdir = pathlib.Path(args.workdir).expanduser().resolve()

    with open(workdir / 'hyperparameters.json', 'r') as fp:
        x = json.load(fp)
    cfg = config_dict.ConfigDict(x)

    with open(workdir / 'data_splits.json', 'r') as fp:
        splits = json.load(fp)

    pick_idx = splits[args.on_split] if args.on_split != 'full' else None

    metrics = from_config.run_evaluation(
        config=cfg,
        num_test=args.num_test,
        pick_idx=pick_idx,
        on_split=args.on_split
    )
    print('Metrics are reported in eV and Angstrom.')
    print(metrics)


if __name__ == '__main__':
    evaluate_so3krates_sparse()
