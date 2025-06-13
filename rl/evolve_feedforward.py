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
from game_base import Mark, TERMINAL_REWARDS
from gameplay import Match
from policies import Policy
from result_viz import load_value_func
import time
from neat.six_util import itervalues, iterkeys


class Arena(object):
    def __init__(self, opp_pi):
        self.opp_pi = opp_pi
        self.env = Environment(opponent_policy=self.opp_pi, player_mark=Mark.X)

    def play_matches(self, agent_pi, n_games=10):
        """
        Play a match with the opponent policy.
        :param n_games: The number of games to play.
        :returns: The average reward per game.
        """
        match = Match(agent_pi, self.opp_pi)
        rewards = [TERMINAL_REWARDS[match.play()] for _ in range(n_games)]
        return np.mean(rewards)


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


N_EVAL_GAMES = 50


def get_opponent():

    return MiniMaxPolicy(Mark.O)


def eval_genomes(genomes, config):
    common_arena = [Arena(get_opponent())]  # just make one of these
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        agent = NNetPolicy(net)
        fitness = common_arena[0].play_matches(agent, n_games=N_EVAL_GAMES)
        # print("\tgenome %s: fitness = %.2f" % (genome_id, fitness))
        genome.fitness = fitness


uncommon_arena = [None]



class StdOutReporterWGenomSizes(neat.StdOutReporter):
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
                g_sizes=[len(g.connections) for g in s.members.values()]
                enable_rates = [np.mean([c.enabled for c in g.connections.values()]) for g in s.members.values()]
                en_str = "{:.2f} ({:0.1f})".format(np.mean(enable_rates), np.std(enable_rates))
                n_connect = "{:.1f} ({:0.1f})".format(np.mean(g_sizes), np.std(g_sizes))
                print(
                    "  {: >4}  {: >3}  {: >4}  {: >13}  {: >14}  {: >9}  {: >7}  {: >7}  {: >4}".format(sid, a, n,n_h_nodes, n_connect,en_str, f, af, st))
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


def eval_genome(genome, config):
    arena = uncommon_arena[0] if uncommon_arena[0] is not None else Arena(get_opponent())
    uncommon_arena[0] = arena  # store the arena for later use
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    agent = NNetPolicy(net)
    fitness = arena.play_matches(agent, n_games=N_EVAL_GAMES)
    # print("\tgenome  fitness = %.2f" % (fitness,))
    genome.fitness = fitness
    return genome.fitness  # return the fitness for the parallel evaluator


def run(config_file, n_cores=10):
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    # Create the population, which is the top-level object for a NEAT run.
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(StdOutReporterWGenomSizes(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    pe = neat.ParallelEvaluator(n_cores, eval_genome)

    # Run for up to 300 generations.
    for iter in range(10):
        print("Iteration %d" % (iter*10))

        if n_cores > 1:
            winner = p.run(pe.evaluate, 10)
        else:
            winner = p.run(eval_genomes, 10)

        # Save the winner.
        encoding = NNetPolicy.INPUT_ENC_SIZES[config.genome_config.num_inputs]
        filename = 'neat-checkpoint-%i-%s.pkl' % (iter, encoding)
        with open(filename, 'wb') as f:
            pickle.dump(winner, f)
            print("SAVED WINNER FOR ITER %i:  %s" % (iter, filename))


    #visualize.draw_net(config, winner, view=True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    # p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    logging.basicConfig(level=logging.INFO)
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    run(config_path)
