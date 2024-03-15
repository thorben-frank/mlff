import argparse
import json
from mlff.config import from_config
from ml_collections import config_dict
import pathlib
import yaml


def train_itp_net():
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
            cfg = config_dict.ConfigDict(yaml.load(fp, Loader=yaml.FullLoader))

    from_config.run_training(cfg, model='itp_net')


if __name__ == '__main__':
    train_itp_net()
