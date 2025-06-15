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
import argparse
from neat.six_util import itervalues, iterkeys

NEAT_REWARDS = {Mark.X: {Result.X_WIN: 1.0, Result.O_WIN: -1.0, Result.DRAW: 0.0},
                Mark.O: {Result.X_WIN: -1.0, Result.O_WIN: 1.0, Result.DRAW: 0.0}}


NETWORK_DIR ='NEAT_nets'
CONFIG_FILE = 'config-feedforward'



N_EVAL_GAMES = 150  # games to play for evaluating a genome (half first, half second)

# strong/weak, input encoding, pop size , generation
WINNER_FILE = f'neat-winner-%s_in=%s_p=%i_neval={N_EVAL_GAMES}_gen=%i.pkl'
# input encoding, pop size  (generation number will be added automatically)
POPULATION_PREFIX = f'neat-population-%s_in=%s_p=%i_neval={N_EVAL_GAMES}_gen='  # strong/weak, input encoding, pop size



class Arena(object):
    def __init__(self, opp_pi, strong=False):
        """
        :param opp_pi: The opponent policy to play against.
        :param strong: If True, we are learing a STRONG solution, starting from any possible state.
            If false, it's the WEAK solution, starting from one of the 10 initial states.
        """
        self.strong = strong
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
        
        self._states_going_first, self._states_going_second = self._get_all_updatable_states()

    def play_matches(self, agent_pi, n_games=10):
        """
        Play a match with the opponent policy.
        :param n_games: The number of games to play.
        :param agent_pi: The policy of the agent to play against the opponent.

        :returns: The average reward per game.
        """
        match = Match(agent_pi, self.opp_pi)
        if not self.strong:
            # if we are learning a weak solution, start from one of the initial states
            traces = [match.play_and_trace(order=-1) for _ in range(n_games//2)] +\
                [match.play_and_trace(order=1) for _ in range(n_games//2)]
        else:

            first_sample = np.random.choice(self._states_going_first, n_games//2, replace=False)
            second_sample = np.random.choice(self._states_going_second, n_games//2, replace=False)
            traces = [match.play_and_trace(order=-1, initial_state=state) for state in first_sample] +\
                     [match.play_and_trace(order=1, initial_state=state) for state in second_sample]
        rewards = self._score_games(traces)
        return np.mean(rewards)

    def _get_all_updatable_states(self):
        logging.info("Getting all updatable states from the environment, sorting by who went first...")
        updatable = self.env.get_nonterminal_states()
        going_first = [state for state in updatable if state.n_free() % 2 == 0]
        going_second = [state for state in updatable if state.n_free() % 2 == 1]
        logging.info("\tFound %d updatable states, %d going first, %d going second" %
                     (len(updatable), len(going_first), len(going_second)))
        return going_first, going_second
    
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
    parser = argparse.ArgumentParser(description="Run NEAT evolution for Tic Tac Toe.")
    parser.add_argument('-c', '--n_cores', type=int, default=12,
                        help='Number of cores to use for parallel evaluation (default: 12)')
    parser.add_argument('-g','--generations', type=int, default=100, help='Number of generations to run (default: 100)')
    parser.add_argument('-p', '--pop_size', type=int, default=None,
                        help='Population size, overrides config file (default: [config file value])')
    parser.add_argument('-r', '--resume_population', type=str, default=None,
                        help='Path to a population file to resume from (default: None, start fresh)')
    parser.add_argument('-s', '--strong', action='store_true',
                        help='If set, learn to maximize reward from any state, otherwise learn from initial states only')
    args = parser.parse_args()

    net_dir = os.path.join(os.getcwd(), 'NEAT_nets')
    config_path = os.path.join(os.getcwd(), CONFIG_FILE)

    net_dir = os.path.join(os.getcwd(), NETWORK_DIR)
    if not os.path.exists(net_dir):
        logging.info("Creating directory for NEAT networks: %s" % net_dir)
        os.mkdir(net_dir)
    else:
        logging.info("Using existing directory for NEAT networks: %s" % net_dir)

    pop_file = os.path.join(os.getcwd(), sys.argv[2]) if len(sys.argv) > 2 else None
    return {'config_file': config_path,
            'generations': args.generations,
            'pop_file': pop_file,
            'pop_size': args.pop_size,
            'n_cores': args.n_cores,
            'strong': args.strong,
            'pop_file': args.resume_population,
            'network_dir': net_dir}

def run():

    args = _get_args()

    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         args['config_file'])
    if args['pop_size'] is not None:
        # Override the population size in the configuration.
        config.pop_size = args['pop_size']
        print("------>  Population overriding config file value, size set to %d" % config.pop_size)
    

    # Create the population, which is the top-level object for a NEAT run.
    if args['pop_file'] is not None:
        # Load a population from a file.
        print("------>  Loading population from file: %s" % args['pop_file'])

        p = neat.Checkpointer.restore_checkpoint(args['pop_file'])
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
    strongweak_string = 'strong' if args['strong'] else 'weak'

    encoding = NNetPolicy.INPUT_ENC_SIZES[config.genome_config.num_inputs]
    chk_file_prefix = os.path.join(os.getcwd(),NETWORK_DIR, POPULATION_PREFIX % (strongweak_string,encoding, config.pop_size))
    checkpointer = neat.Checkpointer(generation_interval=None,   # do this manually
                                     time_interval_seconds=None,
                                     filename_prefix=chk_file_prefix)
    p.add_reporter(checkpointer)

    # Add a stdout reporter to show progress in the terminal.
    stats = neat.StatisticsReporter()
    reporters = [StdOutReporterWGenomSizesPlus(True), stats]
    for r in reporters:
        p.add_reporter(r)

    pe = neat.ParallelEvaluator(args['n_cores'], eval_genome_gameplay) if args['n_cores'] > 1 else None

    pop_backup_interval = 10  # generations between population backups
    for iter in range(args['generations']):
        
        print("Iteration %d (pop=%i, evals=%i, cores=%i)" % (iter, config.pop_size,
                                                            N_EVAL_GAMES, args['n_cores']))


        if args['n_cores'] > 1:
            winner = p.run(pe.evaluate, 1)
        else:
            winner = p.run(eval_genomes_gameplay, 1)

        # Save the winner.
        encoding = NNetPolicy.INPUT_ENC_SIZES[config.genome_config.num_inputs]
        filename = os.path.join(os.getcwd(),NETWORK_DIR, WINNER_FILE % (strongweak_string,encoding, config.pop_size, p.generation))
        with open(filename, 'wb') as f:
            pickle.dump(winner, f)
            print("--------> Saved winner genome, gen %i:  %s" % (p.generation, filename))
            
        if (iter +1) % pop_backup_interval == 0 or iter == args['generations'] - 1:
            checkpointer.save_checkpoint(config=config,
                                        population=p.population,
                                        species_set=p.species,
                                        generation=p.generation)
            print("----------> Save population checkpoint: %s" % (filename,))

    # visualize.draw_net(config, winner, view=True)
    visualize.plot_stats(stats, ylog=False, view=True, pop_size=config.pop_size)
    visualize.plot_species(stats, view=True, pop_size=config.pop_size)

    # p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-4')
    # p.run(eval_genomes, 10)
    # Save population for continuation.


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    logging.basicConfig(level=logging.INFO)
    run()
