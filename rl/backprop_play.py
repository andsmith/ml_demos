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


def play_minimax(n_games, n_hidden, n_epochs, w_alpha, encoding='one-hot'):
    img_size = (1590, 980)

    agent = train_net(n_hidden=n_hidden, n_epochs=n_epochs, encoding=encoding, w_alpha=w_alpha)
    opponent = MiniMaxPolicy(player_mark=Mark.O)
    viz = PolicyEvaluationResultViz(player_policy=agent, opp_policy=opponent)

    viz.play(n_games=n_games)
    img = viz.draw(img_size)
    cv2.imshow("NEAT vs MiniMax", img[:, :, ::-1])
    cv2.waitKey(0)


def get_args():
    parser = argparse.ArgumentParser(description="Play NEAT vs MiniMax")
    parser.add_argument('n_hidden', type=int,
                        help='Number of hidden units learning the policy.')
    parser.add_argument('-w', '--weight_alpha', type=float, default=0.0,
                        help='Weigh early moves more, w=(n_free)^weight_alpha, default off.')
    parser.add_argument('-n', '--n_games', type=int, default=2000,
                        help='Number of games to play.')
    parser.add_argument('-e','--encoding', type=str, default='one-hot',
                        help='Encoding for the neural network input. Options: "enc", "enc+free", "one-hot".')
    parser.add_argument('-i', '--n_epochs', type=int, default=5000,)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = get_args()
    play_minimax(args.n_games, args.n_hidden, n_epochs=args.n_epochs, encoding=args.encoding, w_alpha=args.weight_alpha)



    """
    backprop_play:

    Notes:  Show hidden=0 first, 3 encodings, w=0.0, 1.0, 2.0.

    Show hiden=18, enc+hot, w=0, w=2

    Show hiden 36 with/without weights
        python .\backprop_play.py 36 -e enc+free -w 2.0  -i 700
        
    Show hidden 150
    
    
    
    
    """