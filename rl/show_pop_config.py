import neat
import pprint
import os
import sys
from evolve_feedforward import StdOutReporterWGenomSizesPlus
import logging
import numpy as np
import pickle

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


def _show_pop_config(filename):
    pop = neat.Checkpointer.restore_checkpoint(filename)
    config = pop.config
    print("========================================")
    print(f"Configuration for NEAT population file: {filename}")
    print("========================================")
    pprint.pprint(get_config(config))
    print("========================================")
    
def _show_genome_config(filename):
    """
    Read the genome, print the number of input, output, hidden nodes,
    print the number of connections, then fraction that are enabled.
    """
    with open(filename,'rb') as infile:
        genome = pickle.load(infile)
    
    print("========================================")
    print(f"Configuration for NEAT Genome: {filename}")
    print("========================================")
    #import ipdb; ipdb.set_trace()
    connection_origins = [connect_key[0] for connect_key in genome.connections]
    connection_dests = [node_key for node_key in genome.nodes]
    n_inputs = -np.min(connection_origins)
    # find the highest consecutive node id
    # import ipdb; ipdb.set_trace()
    breaks = np.diff(connection_dests)
    if np.all(breaks==1) :
        n_outputs = len(connection_dests) 
        n_hidden=0
        hidden_conns = []
        hidden_ids = []
    else:
        n_outputs = np.where(breaks>1)[0][0] + 1
        n_hidden = len(connection_dests )-n_outputs
        hidden_ids = [node_id for node_id in connection_dests if node_id >= n_outputs]
        hidden_conns = [conn for conn in genome.connections if conn[0] in hidden_ids or conn[1] in hidden_ids]
    import ipdb; ipdb.set_trace()
    print(f"Number of input nodes:  {n_inputs}")
    print(f"Number of output nodes: {n_outputs}")
    print(f"Number of hidden nodes: {n_hidden}")
    print(f"Number of total connections: {len(genome.connections)}")
    print(f"Number of hidden connections: {len(hidden_conns)}")
    enabled_connections = sum(1 for conn in genome.connections if genome.connections[conn].enabled)
    print(f"Enabled connections: {enabled_connections} ({enabled_connections / len(genome.connections):.2%})")
    print("========================================")
    


def show_config():
    try:
        filename = sys.argv[1]
    except Exception:
        print("Usage: python show_config.py <population_file/winner_file>")
        sys.exit(1)
    if not os.path.exists(filename):
        print(f"File {filename} does not exist.")
        sys.exit(1)
    if filename.endswith('.pkl'):
        _show_genome_config(filename)
    else:
        _show_pop_config(filename)


if __name__== "__main__":
    logging.basicConfig(level=logging.INFO)
    show_config()