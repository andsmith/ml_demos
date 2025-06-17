from gameplay import Match
import pickle
import numpy as np
from result_viz import PolicyEvaluationResultViz
import logging
from game_base import Mark
import cv2
from evolve_feedforward import Arena, NNetPolicy
from perfect_player import MiniMaxPolicy
from baseline_players import HeuristicPlayer
import neat
import sys
import argparse
from evolve_feedforward import NETWORK_DIR
import os
config_file = os.path.join(os.getcwd(), 'config-feedforward')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_file)


def load_nnet_policy(filename):
    with open(filename, 'rb') as f:
        winner_genome = pickle.load(f)
    net = neat.nn.FeedForwardNetwork.create(winner_genome, config)

    agent = NNetPolicy(net)
    return agent


def neat_play_minimax(n_games, filename):
    img_size = (1590, 980)
    agent = load_nnet_policy(filename)
    opponent = MiniMaxPolicy(player_mark=Mark.O)
    viz = PolicyEvaluationResultViz(player_policy=agent, opp_policy=opponent)
    viz.play(n_games=n_games)
    img = viz.draw(img_size)
    cv2.imshow("NEAT vs MiniMax", img[:, :, ::-1])
    cv2.waitKey(0)


def get_args():
    parser = argparse.ArgumentParser(description="Play NEAT vs MiniMax")
    parser.add_argument('genome_file', type=str,
                        help='Path to the genome file (e.g., NEAT_nets/neat-checkpoint-19-enc+free.pkl)')
    parser.add_argument('-n', '--n_games', type=int, default=2000,
                        help='Number of games to play.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = get_args()
    neat_play_minimax(args.n_games, args.genome_file)