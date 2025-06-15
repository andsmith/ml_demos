"""
Adapted from neat/examples/Xor/evolve_feedforward.py
"""

from __future__ import print_function
import os
import neat
import visualize
import numpy as np
import pickle
import logging
from reinforcement_base import Environment
from perfect_player import MiniMaxPolicy
from game_base import Mark, Result, OTHER_GUY, WIN_MARKS
from gameplay import Match
from policies import Policy
from result_viz import load_value_func
import sys
import time
from neat.six_util import itervalues, iterkeys

NEAT_REWARDS = {Mark.X: {Result.X_WIN: 1.0, Result.O_WIN: -1.0, Result.DRAW: 0.0},
                Mark.O: {Result.X_WIN: -1.0, Result.O_WIN: 1.0, Result.DRAW: 0.0}}

NETWORK_DIR = os.path.join(os.getcwd(), 'NEAT_nets')
CONFIG_FILE = 'config-feedforward'


N_EVAL_GAMES = 101  # games to play for evaluating a genome (100 going first, 100 going second)

WINNER_FILE = f'neat-winner_in=%s_p=%i_neval={N_EVAL_GAMES}_gen=%i.pkl'   # input encoding, pop size , generation
# input encoding, pop size  (generation number will be added automatically)
POPULATION_PREFIX = f'neat-population_in=%s_p=%i_neval={N_EVAL_GAMES}_gen='

class Arena(object):
    def __init__(self, opp_pi):
        self.opp_mark = opp_pi.player  # the opponent's mark is the player of the opponent policy
        self.player_mark = OTHER_GUY[self.opp_mark]  # the player is the opponent's opponent
        self.opp_pi = opp_pi
        self._rewards = NEAT_REWARDS[self.player_mark]  # rewards for the opponent's mark
        self.env = Environment(opponent_policy=self.opp_pi, player_mark=Mark.X)
        # the result that is a loss for the player
        self._loss_result = [result for result, mark in WIN_MARKS.items() if WIN_MARKS[result] == self.opp_mark][0]
        logging.info("Arena initialized with opponent mark: %s, player mark: %s, losing result: %s" % (
                     self.opp_mark.name,
                     self.player_mark.name,
                     self._loss_result.name))

    def play_matches(self, agent_pi, n_games=10):
        """
        Play a match with the opponent policy.
        :param n_games: The number of games to play.
        :returns: The average reward per game.
        """
        match = Match(agent_pi, self.opp_pi)
        traces = [match.play_and_trace(order=-1) for _ in range(n_games//2)] +\
                 [match.play_and_trace(order=1) for _ in range(n_games//2)]
        rewards = self._score_games(traces)
        return np.mean(rewards)

    def _get_n_player_moves(self, trace):
        """
        Get the number of moves made by the player in the trace.
        :param trace: The trace of the game as returned by Match.play_and_trace().
        :returns: The number of moves made by the player.
        """
        player_moves = [turn['player'] == self.player_mark for turn in trace['game']]
        return sum(player_moves)

    def _score_games(self, traces, sliding_loss=True):
        """
        win is scored as 1 
        draw is scored as 0
        loss is scored as -1, unless sliding loss is True:
           - loses after making 2nd move:  reward = -3.0
           - loses after making 3rd move:  reward = -2.0
           - loses after making 4th move:  reward = -1.0  

        NOTE:  Win will not happen using MiniMax Opponent.
        TODO: Add sliding_win for other opponents?

        :param traces:  list of outputs of Match.play_and_trace()
        :param sliding_loss:  if True, adjust the loss penalty based on the number of moves made by the player.
        """
        rewards = np.array([self._rewards[trace['result']] for trace in traces])
        if sliding_loss:
            is_loss = np.array([trace['result'] == self._loss_result for trace in traces])
            # adjust the loss penalty based on the length of the game
            n_moves = np.array([self._get_n_player_moves(trace) for trace in traces])
            sliding_penalty = (5-n_moves)  # 3 for losing after 2nd, etc.
            rewards[is_loss] = -sliding_penalty[is_loss]
        return rewards


class NNetPolicy(Policy):

    INPUT_ENC_SIZES = {9: 'enc',
                       18: 'enc+free',  # 9 + 9 free actions
                       27: 'one-hot'}  # 9 + 18 free actions

    def __init__(self, nnet):
        self._net = nnet
        input_size = len(nnet.input_nodes)
        self.encoding = encoding = self.INPUT_ENC_SIZES[input_size]
        self.player = Mark.X
        if encoding not in ['enc', 'enc+free', 'one-hot']:
            raise ValueError("encoding must be one of 'enc', 'enc+free', or 'one-hot'")

    def recommend_action(self, state):
        """
        Given a state, return the action disrtibution (deterministic for nnet, highest output for free actions).
        :param state: The current state of the game.
        :returns: a list of (action, prob) tuples, where prob is the probability of taking the action.
        """
        inputs = state.to_nnet_input(method=self.encoding)  # get the input for the neural network
        output = np.array(self._net.activate(inputs))
        open_actions = state.get_actions(flat_inds=True)
        valid_outputs = output[open_actions]
        bi = np.argmax(valid_outputs)
        best_action_flat = open_actions[bi]  # get the index of the best action in the flat array
        best_action = (best_action_flat // 3, best_action_flat % 3)  # convert to (i, j) tuple
        return [(best_action, 1.0)]  # return the best action with probability 1.0



def get_opponent():

    return MiniMaxPolicy(Mark.O)


shared_arena = [None]


def eval_genome_gameplay(genome, config):
    arena = shared_arena[0] if shared_arena[0] is not None else Arena(get_opponent())
    shared_arena[0] = arena  # store the arena for the rest of the genomes
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    agent = NNetPolicy(net)
    fitness = arena.play_matches(agent, n_games=N_EVAL_GAMES)
    genome.fitness = fitness
    return genome.fitness  # return the fitness for the parallel evaluator


def eval_genomes_gameplay(genomes, config):
    for genome_id, genome in genomes:
        eval_genome_gameplay(genome, config)


class StdOutReporterWGenomSizesPlus(neat.StdOutReporter):
    """A reporter that outputs the genome sizes along with the standard output."""

    def end_generation(self, config, population, species_set):
        # Modified copy of neat.reporting.StdOutReporter.end_generation
        self.generation += 1
        ng = len(population)
        ns = len(species_set.species)
        n_output = config.genome_config.num_outputs
        if self.show_species_detail:
            print('Population of {0:d} members in {1:d} species:'.format(ng, ns))
            sids = list(iterkeys(species_set.species))
            sids.sort()
            print("    ID  age   pop  mean # hidden    mean # edges   m enabled  fitness  adj fit  stag")
            print("  ====  ===  ====  =============  ==============  ==========  =======  =======  =====")
            for sid in sids:
                s = species_set.species[sid]
                a = self.generation - s.created
                n = len(s.members)
                f = "--" if s.fitness is None else "{:.1f}".format(s.fitness)
                af = "--" if s.adjusted_fitness is None else "{:.3f}".format(s.adjusted_fitness)
                st = self.generation - s.last_improved
                hid_sizes = [len(g.nodes)-n_output for g in s.members.values()]
                n_h_nodes = "{:.2f} ({:0.1f})".format(np.mean(hid_sizes), np.std(hid_sizes))
                g_sizes = [len(g.connections) for g in s.members.values()]
                enable_rates = [np.mean([c.enabled for c in g.connections.values()]) for g in s.members.values()]
                en_str = "{:.2f} ({:0.1f})".format(np.mean(enable_rates), np.std(enable_rates))
                n_connect = "{:.1f} ({:0.1f})".format(np.mean(g_sizes), np.std(g_sizes))
                print(
                    "  {: >4}  {: >3}  {: >4}  {: >13}  {: >14}  {: >9}  {: >7}  {: >7}  {: >4}".format(sid, a, n, n_h_nodes, n_connect, en_str, f, af, st))
        else:
            print('Population of {0:d} members in {1:d} species'.format(ng, ns))

        elapsed = time.time() - self.generation_start_time
        self.generation_times.append(elapsed)
        self.generation_times = self.generation_times[-10:]
        average = sum(self.generation_times) / len(self.generation_times)
        print('Total extinctions: {0:d}'.format(self.num_extinctions))
        if len(self.generation_times) > 1:
            print("Generation time: {0:.3f} sec ({1:.3f} average)".format(elapsed, average))
        else:
            print("Generation time: {0:.3f} sec".format(elapsed))


def _get_args():
    if not os.path.exists(NETWORK_DIR):
        logging.info("Creating directory for NEAT networks: %s" % NETWORK_DIR)
        os.mkdir(NETWORK_DIR)
    else:
        logging.info("Using existing directory for NEAT networks: %s" % NETWORK_DIR)
    config_path = os.path.join(os.getcwd(), CONFIG_FILE)
    n_cores = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    pop_file = os.path.join(os.getcwd(), sys.argv[2]) if len(sys.argv) > 2 else None
    return config_path, pop_file, n_cores


def run():

    config_file, pop_file, n_cores = _get_args()

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    # Create the population, which is the top-level object for a NEAT run.
    if pop_file is not None:
        # Load a population from a file.
        print("------>  Loading population from file: %s" % pop_file)

        p = neat.Checkpointer.restore_checkpoint(pop_file)
        config = p.config  # use the config from the loaded population

        if config.genome_config.num_inputs not in NNetPolicy.INPUT_ENC_SIZES:
            raise ValueError("Invalid number of inputs in the configuration: %d" % config.genome_config.num_inputs)
        print("------>  Population loaded with %d genomes" % len(p.population))
        print("------>  Population encoding: %s" % NNetPolicy.INPUT_ENC_SIZES[config.genome_config.num_inputs])
        print("------>  Population generation: %d" % p.generation)
        print("------>  Population species: %d" % len(p.species.species))
        print("------>  Population size: %d" % config.pop_size)
        print("------>  Population fitness criterion: %s" % config.fitness_criterion)
        print("------>  Population fitness threshold: %s" % config.fitness_threshold)
    else:
        # Create a new population.
        print("------>  Creating new population")
        p = neat.Population(config)

    # add checkpointing reporter

    encoding = NNetPolicy.INPUT_ENC_SIZES[config.genome_config.num_inputs]
    chk_file_prefix = os.path.join(NETWORK_DIR, POPULATION_PREFIX % (encoding, config.pop_size))
    checkpointer = neat.Checkpointer(generation_interval=None,   # do this manually
                                     time_interval_seconds=None,
                                     filename_prefix=chk_file_prefix)
    p.add_reporter(checkpointer)

    # Add a stdout reporter to show progress in the terminal.
    stats = neat.StatisticsReporter()
    reporters = [StdOutReporterWGenomSizesPlus(True), stats]
    for r in reporters:
        p.add_reporter(r)
    
    pe = neat.ParallelEvaluator(n_cores, eval_genome_gameplay) if n_cores > 1 else None

    n_gen_per_iter = 10
    for iter in range(10):
        print("------>  Iteration %d" % (iter*10))

        if n_cores > 1:
            winner = p.run(pe.evaluate, n_gen_per_iter)
        else:
            winner = p.run(eval_genomes_gameplay, n_gen_per_iter)

        # Save the winner.
        encoding = NNetPolicy.INPUT_ENC_SIZES[config.genome_config.num_inputs]
        filename = os.path.join(NETWORK_DIR, WINNER_FILE % (encoding, config.pop_size, p.generation))
        with open(filename, 'wb') as f:
            pickle.dump(winner, f)
            print("------------------------------------->   SAVED WINNER FOR ITER %i:  %s" % (p.generation, filename))

        if iter % 5 == 0:
           # if False:  # TODO: FIX
            species_set = p.species
            generation = p.generation
            checkpointer.save_checkpoint(config=config,
                                         population=p.population,
                                         species_set=species_set,
                                         generation=generation)
            print("----------------------------------------->   SAVED POPULATION FOR ITER %i:  %s" % (iter, filename))

    # visualize.draw_net(config, winner, view=True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    # p.run(eval_genomes, 10)
    # Save population for continuation.


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    logging.basicConfig(level=logging.INFO)
    run()
