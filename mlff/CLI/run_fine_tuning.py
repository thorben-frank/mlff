import argparse
import json
from mlff.config import from_config
from ml_collections import config_dict
import pathlib
import yaml


def fine_tune_so3krates_sparse():
    # Create the parser
    parser = argparse.ArgumentParser(description='Train a SO3kratesSparse model.')
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to the config file.'
    )
    parser.add_argument(
        '--start_from_workdir',
        type=str,
        required=True,
        help='Path to workdir from which fine tuning should be started.'
    )

    parser.add_argument(
        '--strategy',
        type=str,
        required=True,
        help='Path to workdir from which fine tuning should be started.'
    )

    args = parser.parse_args()

    config = pathlib.Path(args.config).expanduser().absolute().resolve()
    if config.suffix == '.json':
        with open(config, mode='r') as fp:
            cfg = config_dict.ConfigDict(json.load(fp=fp))
    elif config.suffix == '.yaml':
        with open(config, mode='r') as fp:
            cfg = config_dict.ConfigDict(yaml.load(fp, Loader=yaml.FullLoader))

    from_config.run_fine_tuning(
        cfg,
        start_from_workdir=args.start_from_workdir,
        strategy=args.strategy
    )


if __name__ == '__main__':
    fine_tune_so3krates_sparse()
