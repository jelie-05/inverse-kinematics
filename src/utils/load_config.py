import argparse
import yaml
from dataclasses import is_dataclass, fields
from utils.config_base import ExperimentConfig

def parse_args():
    parser = argparse.ArgumentParser(description="Load configuration from a YAML file.")
    
    # Add an argument for the configuration file
    parser.add_argument('--config', type=str, help='Path to the configuration file', required=True)
    parser.add_argument('--local-rank', default=-1, type=int, help='Local rank for distributed training')

    return parser.parse_args()

def load_from_yaml(path):
    with open(path, 'r') as file:
        return yaml.safe_load(file)
    
def dict_to_class(cls, d):
    if not is_dataclass(cls):
        return d
    field_types = {f.name: f.type for f in fields(cls)}
    return cls(**{
        k: dict_to_class(field_types[k], v) if k in field_types else v
        for k, v in d.items()
    })
    
def load_config_from_args():
    """
    Load configuration from YAML file and command line arguments.
    Returns:
        config (dict): Configuration dictionary.
        args (argparse.Namespace): Parsed command line arguments."""

    args = parse_args()
    config = load_from_yaml(args.config)
    config = dict_to_class(ExperimentConfig, config)

    return config