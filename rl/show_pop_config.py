import neat
import pprint
import os
import sys
from evolve_feedforward import StdOutReporterWGenomSizesPlus
import logging


def get_config(obj):
    config = {}
    for par in obj.__dict__:
        if par.startswith('_'):
            continue
        value = obj.__dict__[par]
        if hasattr(value, '__dict__'):
            value = get_config(value)
        config[par] = value    
    return config


def _show_config(filename):
    pop = neat.Checkpointer.restore_checkpoint(filename)
    config = pop.config
    print("Configuration for NEAT:")
    print("========================================")
    print(f"Population file: {filename}")
    print("========================================")
    print("Configuration parameters:")
    print("========================================")
    pprint.pprint(get_config(config))
    print("========================================")
    


def show_config():
    try:
        filename = sys.argv[1]
    except Exception:
        print("Usage: python show_config.py <population_file>")
        sys.exit(1)
    if not os.path.exists(filename):
        print(f"File {filename} does not exist.")
        sys.exit(1)
    _show_config(filename)


if __name__== "__main__":
    logging.basicConfig(level=logging.INFO)
    show_config()