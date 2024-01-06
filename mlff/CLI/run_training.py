import argparse
import json
from mlff.config import from_config
from ml_collections import config_dict
import pathlib
import yaml


def train_so3krates_sparse():
    # Create the parser
    parser = argparse.ArgumentParser(description='Train a SO3kratesSparse model.')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')

    args = parser.parse_args()

    config = pathlib.Path(args.config).expanduser().absolute().resolve()
    if config.suffix == '.json':
        with open(config, mode='r') as fp:
            cfg = config_dict.ConfigDict(json.load(fp=fp))
    elif config.suffix == '.yaml':
        with open(config, mode='r') as fp:
            cfg = config_dict.ConfigDict(yaml.load(fp))

    from_config.run_training(cfg)


if __name__ == '__main__':
    train_so3krates_sparse()
