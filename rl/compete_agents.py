from gameplay import Match
import pickle
import numpy as np
from result_viz import PolicyEvaluationResultViz
import logging
from game_base import Mark
import cv2
from perfect_player import MiniMaxPolicy
from backprop_net import train_net
import argparse

from policies import InvPolicy
import neat
from neat_play import load_nnet_policy
from backprop_net import BackpropPolicy

POLICY_FACTORIES = {
    # should match args in get_args() to load properly
    'neat_nets': load_nnet_policy,
    'backprop_nets': BackpropPolicy.from_file
}


def _compete(player1_pi, player2_pi, n_games=1000):
    """
    Compete two policies against each other, return the results.
    :param player1_pi:  first policy to compete.
    :param player2_pi:  second policy to compete.
    :param n_games:  number of games to play.
    :return:  ResultSet object with the results of the competition.
    """
    img_size = (1080, 980)
    if player1_pi.player == player2_pi.player:
        player2_pi = InvPolicy(player2_pi)
    viz = PolicyEvaluationResultViz(player_policy=player1_pi, opp_policy=player2_pi)

    viz.play(n_games=n_games)
    img = viz.draw(img_size)
    cv2.imshow("NEAT vs MiniMax", img[:, :, ::-1])
    cv2.waitKey(0)


def load_policies(player_files):
    players = []
    for pol_key, pol_class in POLICY_FACTORIES.items():
        if pol_key in player_files:
            for filename in player_files[pol_key]:
                players.append(pol_class(filename))
    return players


def run(args):
    player_files = {pol_key: getattr(args, pol_key, None)  # get the list of files for each policy type
                    for pol_key in POLICY_FACTORIES if getattr(args, pol_key) is not None}
    players = load_policies(player_files)
    _compete(players[0], players[1], n_games=args.n_games)



def get_args():
    parser = argparse.ArgumentParser(description="Play NEAT vs MiniMax")
    parser.add_argument('-n', '--neat_nets', type=str, nargs='+',
                        help='NEAT genome file(s) to compete."')
    parser.add_argument('-b', '--backprop_nets', type=str, nargs='+',
                        help='Backpropagation network file(s) to compete."')
    parser.add_argument('-g', '--n_games', type=int, default=2000,
                        help='Number of games to play.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = get_args()

    run(args)

    """
    backprop_play:

    Notes:  Show hidden=0 first, 3 encodings, w=0.0, 1.0, 2.0.

    Show hiden=18, enc+hot, w=0, w=2

    Show hiden 36 with/without weights
        python .\backprop_play.py 36 -e enc+free -w 2.0  -i 700
        
    Show hidden 150
    
    
    
    
    """
