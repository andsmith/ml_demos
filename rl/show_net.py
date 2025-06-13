import pickle
from tic_tac_toe import Game
from evolve_feedforward import NNetPolicy, Arena
import neat
import visualize

import sys
def load(filename):
    with open(filename,'rb') as infile:
        genome = pickle.load(infile)
    return genome

def show(genome):
        
    config_file = 'config-feedforward'
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                            config_file)
    
    visualize.draw_ttt_net(config, genome, view=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python show_net.py <genome_file.pkl>")
        sys.exit(1)

    genome_file = sys.argv[1]
    genome = load(genome_file)
    
    show(genome)