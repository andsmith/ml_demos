from gameplay import Match
import pickle
import numpy as np
from result_viz import PolicyEvaluationResultViz
import logging
from game_base import Mark
import cv2
from evolve_feedforward import Arena, NNetPolicy

from baseline_players import HeuristicPlayer
import neat
config_file = 'config-feedforward'
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_file)



def load_nnet_policy(filename = "neat-checkpoint_-one-hot.pkl"):
    with open(filename,'rb') as f:
        winner_genome = pickle.load(f)
    net = neat.nn.FeedForwardNetwork.create(winner_genome, config)

    agent = NNetPolicy(net)
    return agent

def neat_play(n_rules_2=2):
    img_size = (1590, 980)
    agent = load_nnet_policy()

    opponent = HeuristicPlayer(n_rules=n_rules_2, mark=Mark.O)
    viz = PolicyEvaluationResultViz(player_policy=agent, opp_policy=opponent)
    viz.play(n_games=500)


    img = viz.draw(img_size)
    cv2.imshow("Policy Evaluation Visualization", img[:, :, ::-1])
    cv2.waitKey(0)



if __name__=="__main__":

    logging.basicConfig(level=logging.INFO)
    neat_play()