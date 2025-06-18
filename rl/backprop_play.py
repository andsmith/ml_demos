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


def play_minimax(n_games, n_hidden, n_epochs):
    img_size = (1590, 980)
    agent = train_net(n_hidden=n_hidden, n_epochs=n_epochs)
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
    parser.add_argument('-n', '--n_games', type=int, default=2000,
                        help='Number of games to play.')
    parser.add_argument('-e', '--n_epochs', type=int, default=5000,)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = get_args()
    play_minimax(args.n_games, args.n_hidden, n_epochs=args.n_epochs)