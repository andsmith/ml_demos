"""
Adapted from neat/examples/Xor/evolve_feedforward.py
"""

from __future__ import print_function
import os
import neat
import neat.parallel
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
from neat_util import NNetPolicy, StdOutReporterWGenomSizesPlus, ParallelEvaluatorWStats
from visualize import Visualizer
import time
import argparse
from neat.six_util import itervalues, iterkeys
from checkpoint import CheckpointerWithStats
import matplotlib.pyplot as plt
NEAT_REWARDS = {Mark.X: {Result.X_WIN: 1.0, Result.O_WIN: -1.0, Result.DRAW: 0.0},
                Mark.O: {Result.X_WIN: -1.0, Result.O_WIN: 1.0, Result.DRAW: 0.0}}


NETWORK_DIR = 'NEAT_nets'
CONFIG_FILE = 'config-gameplay'
CONFIG_TEACHER_FILE = 'config-teacher'

# strong/weak, input encoding, pop size , generation
WINNER_FILE = f'neat-winner-%s_in=%s_p=%i_neval=%i_gen=%i.pkl'
# input encoding, pop size  (generation number will be added automatically)
POPULATION_PREFIX = f'neat-population-%s_in=%s_p=%i_neval=%i_gen='  # strong/weak, input encoding, pop size


class Arena(object):
    """
    Evaluate genomes by playing games against another policy.
    """

    def __init__(self, opp_pi, strong=False, sliding_loss=False):
        """
        :param opp_pi: The opponent policy to play against.
        :param strong: If True, we are learing a STRONG solution, starting from any possible state.
            If false, it's the WEAK solution, starting from one of the 10 initial states.
        """
        self.strong = strong
        self.sliding_loss = sliding_loss  # if True, use sliding loss penalty
        self.opp_mark = opp_pi.player  # the opponent's mark is the player of the opponent policy
        self.player_mark = OTHER_GUY[self.opp_mark]  # the player is the opponent's opponent
        self.opp_pi = opp_pi
        self._rewards = NEAT_REWARDS[self.player_mark]  # rewards for the opponent's mark
        self.env = Environment(opponent_policy=self.opp_pi, player_mark=Mark.X)
        # the result that is a loss for the player
        self._loss_result = [result for result, mark in WIN_MARKS.items() if WIN_MARKS[result] == self.opp_mark][0]
        logging.info("Arena initialized with opponent mark: %s, player mark: %s, strong_solution: %s,  sliding_loss: %s" % (
                     self.opp_mark.name,
                     self.player_mark.name,
                     self.strong,
                     self.sliding_loss))

        self._states_by_turn = self._get_all_updatable_states()

    def play_matches(self, agent_pi, n_games, n_games_min=50, repeat_timeout=9):
        """
        Play a match with the opponent policy.

        if starting from an initial state, repeats are likely, so then, just play UP TO the number specified by n_games.


        :param n_games: The number of games to play.
        :param agent_pi: The policy of the agent to play against the opponent.
        :param repeat_timeout: when this many duplicate games are played in a row, stop playing & return the average reward.
            (Only relevant for weak solutions, where the agent starts from one of the initial states.)
        :returns: The average reward per game.
        """
        def _hash_trace(trace):
            def _hash_state(state):
                return tuple(state.state.flatten().astype(int))
            states = [_hash_state(round['state']) for round in trace['game']]
            states.append(_hash_state(trace['game'][-1]['next_state']))  # add the final state
            return tuple(states)  # return a tuple of states as the hash

        match = Match(agent_pi, self.opp_pi)
        traces = []
        repeat_set = set()
        n_reps = 0
        repeat_states = {'n_matches': 0, 'n_repeats': 0}
        while (repeat_timeout is None and len(traces) < n_games) or \
                (repeat_timeout is not None and n_reps < repeat_timeout and len(traces) < n_games and
                    n_reps < n_games_min):

            index = len(traces)
            order = (-1)**(index)

            if not self.strong:
                new_match = match.play_and_trace(order=order)
            else:
                sample = [self._states_by_turn['first'], self._states_by_turn['second']][index % 2]
                state = np.random.choice(sample, 1)[0]  # pick a random state from the sample
                new_match = match.play_and_trace(order=1, init_state=state)

            trace_hash = _hash_trace(new_match)
            if trace_hash not in repeat_set:
                repeat_set.add(trace_hash)
                repeat_states['n_matches'] += 1
                n_reps = 0
            else:
                repeat_states['n_repeats'] += 1
                n_reps += 1

            traces.append(new_match)  # include all traces in the score, only count new ones for timeout.

        if len(traces) == 0:
            logging.warning("n_games:  %i, n_games_min: %i, repeat_timeout: %i, n_reps: %i, repeat_stats:  %s" %
                            (n_games, n_games_min, repeat_timeout, n_reps, repeat_states))
            raise Exception("No games played.")

        rewards = self._score_games(traces)
        return np.mean(rewards), repeat_states

    def _get_all_updatable_states(self):

        logging.info("Getting all updatable states from the environment, sorting by who went first...")
        updatable = self.env.get_nonterminal_states()
        # Even number of markers, odd number of free cells -> player went first
        going_first = [state for state in updatable if state.n_free() % 2 == 1]
        going_second = [state for state in updatable if state.n_free() % 2 == 0]
        logging.info("\tFound %d updatable states, %d going first, %d going second" %
                     (len(updatable), len(going_first), len(going_second)))
        return {'first': going_first, 'second': going_second}

    _SLIDING_LOSS = {  # for weak solution  (win/draw in self._rewards)
        # went_first: {n_moves: reward, ...}
        True: {4: -2.0,  # going first, losing after 4th move
               3: -4.0},  # going first, lost quickest, most punished
        False: {4: -1.0,  # going second, losing after 4th move, most difficult, least punished
                3: -2.0,  # going second, losing after 3rd move
                2: -3.0}}  # going second, losing after 2nd move

    def _score_games(self, traces):
        """
        win is scored as 1
        draw is scored as 0
        loss is scored as -1, unless self.sliding_loss.

        NOTE:  Win will not happen using MiniMax Opponent, use sliding loss only for weak solvers?
        TODO: Add sliding_win for other opponents?

        :param traces:  list of outputs of Match.play_and_trace()
        :param sliding_loss:  if True, adjust the loss penalty based on the number of moves made by the player.
        """
        rewards = np.array([self._rewards[trace['result']] for trace in traces])
        if len(traces) < 2:
            logging.warning("Not enough traces to score games, returning -5.0 as fitness.")
            return np.ones_like(rewards) * -5  # degenerate cases have worst fitness

        if self.sliding_loss:
            is_loss = [trace['result'] == self._loss_result for trace in traces]
            firstp_and_n_moves = np.array([self._get_moves(trace)
                                          for i, trace in enumerate(traces) if is_loss[i]]).reshape(-1, 2)
            n_moves = firstp_and_n_moves[:, 1]  # number of moves made by the player
            went_first = np.array(firstp_and_n_moves)[:, 0]  # whether the player went first

            sliding_losses = np.array([self._SLIDING_LOSS[went_first[i]][n_moves[i]] for i in range(len(n_moves))])
            rewards[is_loss] = np.array(sliding_losses)
        return rewards

    def _get_moves(self, trace):
        """
        Determine if player went first.
        Determine number of moves player made.

        NOTE:  Trace might not start from empty state or from player's turn.

        :param trace: The trace of the game as returned by Match.play_and_trace().
        :returns: (True if player went first, number of moves made by the player)
        """
        player_turn = [round['player'] == self.player_mark for round in trace['game']]
        first_player_turn = np.where(player_turn)[0][0]  # index of the first player's turn
        first_state = trace['game'][first_player_turn]['state']
        n_marks = 9 - first_state.n_free()
        went_first = (n_marks % 2 == 0)

        final_state = trace['game'][-1]['next_state']
        player_moves = np.sum(final_state.state == self.player_mark)
        return (went_first, player_moves)


class Teacher(Arena):
    def __init__(self, teacher_pi):
        self._pi = teacher_pi  # the policy to use as a teacher
        self.player = self._pi.player  # the player's mark is the same as the teacher's mark
        self.env = Environment(opponent_policy=self._pi, player_mark=self.player)
        self._states_by_turn = self._get_all_updatable_states()

    def eval_network(self, net_pi, n_states=1000):
        """
        See if the network chooses one of the teacher's recommended actions for a sample of n_states states.
        :param net_pi: A NNetPolicy instance with the network to evaluate.
        :param n_states: The number of states to sample.
        :returns: The fraction of states where the network chose one of the teacher's recommended actions.
        """
        n_first = min(n_states // 2, len(self._states_by_turn['first']))
        n_second = min(n_states - n_first, len(self._states_by_turn['second']))
        going_first_states = np.random.choice(self._states_by_turn['first'], n_first, replace=False).tolist()
        going_second_states = np.random.choice(self._states_by_turn['second'], n_second, replace=False).tolist()
        good_choice = [self._eval_decision(net_pi, state) for state in going_first_states + going_second_states]
        return np.mean(good_choice), (n_first, n_second)

    def _eval_decision(self, net_pi, state):
        net_choice = net_pi.recommend_action(state)[0][0]
        teacher_choices = [action for action, _ in self._pi.recommend_action(state)]
        was_good = net_choice in teacher_choices
        return was_good

# Baseline player policies:


def get_opponent():
    return MiniMaxPolicy(Mark.O)


def get_teacher():
    return MiniMaxPolicy(Mark.X)


# Shared state for multiprocessing pool:
shared_stats = [None]
shared_arena = [None]


def eval_genome_gameplay(genome, config, get_stats=True):
    # Run this in parallel (via the ParallelEvaluatorWStats)
    arena = shared_arena[0] if shared_arena[0] is not None else Arena(
        get_opponent(), strong=config.strong, sliding_loss=config.sliding_loss)
    shared_arena[0] = arena  # store the arena for the rest of the genomes
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    agent = NNetPolicy(net)
    fitness, repeat_stats = arena.play_matches(agent, n_games=config.n_evals)
    genome.fitness = fitness
    if get_stats:
        return genome.fitness, repeat_stats  # return the fitness for the parallel evaluator
    return genome.fitness


def eval_genomes_gameplay(genomes, config):
    # For running serially (via neat.Population.run())
    shared_stats[0] = {'n_matches': [], 'n_repeats': []}  # reset stats for this run
    for genome_id, genome in genomes:
        _, repeat_stats = eval_genome_gameplay(genome, config, get_stats=True)
        shared_stats[0]['n_matches'].append(repeat_stats['n_matches'])
        shared_stats[0]['n_repeats'].append(repeat_stats['n_repeats'])


shared_teacher = [None]


def eval_genome_teacher(genome, config, get_stats=True):
    # Run this in parallel (via the ParallelEvaluatorWStats)
    teacher = shared_teacher[0] if shared_teacher[0] is not None else Teacher(
        get_teacher())
    shared_teacher[0] = teacher  # store the teacher for the rest of the genomes
    agent = NNetPolicy(neat.nn.FeedForwardNetwork.create(genome, config))
    good_choice_fraction, n_train = teacher.eval_network(agent, n_states=config.n_evals)
    genome.fitness = good_choice_fraction
    if get_stats:
        return genome.fitness, {'n_matches': n_train, 'n_repeats':0}
    return genome.fitness


def eval_genomes_teacher(genomes, config):
    shared_stats[0] = {'n_matches': [], 'n_repeats': []}
    for genome_id, genome in genomes:
        eval_genome_teacher(genome, config, get_stats=True)
        shared_stats[0]['n_matches'].append(config.n_evals)


def _get_args():
    parser = argparse.ArgumentParser(description="Run NEAT evolution for Tic Tac Toe:")
    parser.add_argument('-n', '--neat_config', type=str, default=None,
                        help='NEAT config file (default: config-gameplay or config-teacher).')
    parser.add_argument('-c', '--n_cores', type=int, default=12,
                        help='Number of cores to use for parallel evaluation (default: 12).')
    parser.add_argument('-g', '--generations', type=int, default=100,
                        help='Number of generations to run (default: 100).')
    parser.add_argument('-e', '--n_eval', type=int, default=100,
                        help='Number of games to play for evaluating a genome (default: 100).')
    parser.add_argument('-p', '--pop_size', type=int, default=None,
                        help='Population size, overrides config file (default: [config file value]).')
    parser.add_argument('-r', '--resume_population', type=str, default=None,
                        help='Path to a population file to resume from (default: None, start fresh).')
    parser.add_argument('-s', '--strong', action='store_true',
                        help='If set, learn to maximize reward from any state, otherwise learn from initial states only.')
    parser.add_argument('-l', '--sliding_loss', action='store_true', default=True,
                        help='If set, use "sliding" loss penalty, losing in fewer moves is worse (default: True).')
    parser.add_argument('-t', '--teacher', action='store_true', default=False,
                        help='If true, evaluate by comparing w/teacher policy, if false, by playing the opponent policy (default: False).')
    args = parser.parse_args()

    if args.neat_config is None:
        args.neat_config = CONFIG_FILE if not args.teacher else CONFIG_TEACHER_FILE
        
    net_dir = os.path.join(os.getcwd(), 'NEAT_nets')

    net_dir = os.path.join(os.getcwd(), NETWORK_DIR)
    if not os.path.exists(net_dir):
        logging.info("Creating directory for NEAT networks: %s" % net_dir)
        os.mkdir(net_dir)
    else:
        logging.info("Using existing directory for NEAT networks: %s" % net_dir)

    pop_file = os.path.join(os.getcwd(), sys.argv[2]) if len(sys.argv) > 2 else None
    return {'config_file': args.neat_config,
            'sliding_loss': args.sliding_loss,
            'network_dir': net_dir,
            'teacher': args.teacher,
            'generations': args.generations,
            'pop_file': pop_file,
            'pop_size': args.pop_size,
            'n_cores': args.n_cores,
            'n_eval': args.n_eval,
            'strong': args.strong,
            'pop_file': args.resume_population,
            'network_dir': net_dir}


def run():

    args = _get_args()

    # Load configuration.
    logging.info("Loading NEAT configuration from: %s" % args['config_file'])
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         args['config_file'])
    config.n_evals = args['n_eval']  # set the number of evaluations per genome
    config.strong = args['strong']  # set the strong/weak flag
    config.sliding_loss = args['sliding_loss']  # set the sliding loss flag
    config.teacher = args['teacher']  # set the teacher flag

    if args['pop_size'] is not None:
        # Override the population size in the configuration.
        config.pop_size = args['pop_size']
        print("------>  Population overriding config file value, size set to %d" % config.pop_size)

    # Create the population, which is the top-level object for a NEAT run.
    if args['pop_file'] is not None:
        # Load a population from a file.
        print("------>  Loading population from file: %s" % args['pop_file'])
        p = CheckpointerWithStats.restore_checkpoint(args['pop_file'])
        if isinstance(p, tuple):
            p, stats = p
        else:
            logging.warning("Resuming population with no stats file, statistics will be accumulated from this point on.")
            stats = neat.StatisticsReporter()
        
        # override values:    
        config = p.config  # use the config from the loaded population

        # backwards compatibility:
        if not hasattr(config, 'teacher'):
            config.teacher = False

        if config.genome_config.num_inputs not in NNetPolicy.INPUT_ENC_SIZES:
            raise ValueError("Invalid number of inputs in the configuration: %d" % config.genome_config.num_inputs)
        print("------>  Population loaded with %d genomes" % len(p.population))
        print("------>  Population encoding: %s" % NNetPolicy.INPUT_ENC_SIZES[config.genome_config.num_inputs])
        print("------>  Population generation: %d" % p.generation)
        print("------>  Population species: %d" % len(p.species.species))
        print("------>  Population size: %d" % config.pop_size)
        print("------>  Population fitness criterion: %s" % config.fitness_criterion)
        print("------>  Population fitness threshold: %s" % config.fitness_threshold)
        print("------>  n_evals: %d" % config.n_evals)
        print("------>  Training type: %s" %
              'playing against opponent' if not config.teacher else 'comparing with teacher policy')
        if not config.teacher:
            print("------>  Finding strong solution?: %s" % config.strong)
            print("------>  Sliding loss?: %s" % config.sliding_loss)

    else:
        # Create a new population.
        print("------>  Creating new population")
        p = neat.Population(config)
        stats = neat.StatisticsReporter()

    # add checkpointing reporter
    strongweak_string = ('strong' if args['strong'] else 'weak') if not config.teacher else 'mimic'

    encoding = NNetPolicy.INPUT_ENC_SIZES[config.genome_config.num_inputs]
    chk_file_prefix = os.path.join(os.getcwd(), NETWORK_DIR, POPULATION_PREFIX %
                                   (strongweak_string, encoding, config.pop_size, args['n_eval']))
    checkpointer = CheckpointerWithStats(generation_interval=None,   # do this manually
                                         time_interval_seconds=None,
                                         filename_prefix=chk_file_prefix)
    p.add_reporter(checkpointer)
    reporters = [StdOutReporterWGenomSizesPlus(True), stats]
    for r in reporters:
        p.add_reporter(r)

    if config.teacher:
        pe = ParallelEvaluatorWStats(args['n_cores'], eval_genome_teacher) if args['n_cores'] > 1 else None
    else:
        pe = ParallelEvaluatorWStats(args['n_cores'], eval_genome_gameplay) if args['n_cores'] > 1 else None

    # visualizer:
    plt.ion()
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    visualizer = Visualizer(ax, config)
    # plt.show()

    pop_backup_interval = 10  # generations between population backups
    for iter in range(args['generations']):

        print("\tIteration %d (pop=%i, evals=%i, cores=%i)" % (iter, config.pop_size,
                                                               args['n_eval'], args['n_cores']))

        if args['n_cores'] > 1:
            pe.reset_stats()
            winner = p.run(pe.evaluate, 1)
            repeat_stats = pe.stats

        else:

            if config.teacher:
                winner = p.run(eval_genomes_teacher, 1)
            else:
                winner = p.run(eval_genomes_gameplay, 1)

            repeat_stats = shared_stats[0]  # get the stats from the last run

        # Print evaluation repeat statistics:
        print("\tEvaluation played %s (%s) different games, %s (%s) repeats" %
              (np.mean(repeat_stats['n_matches'],axis=0),
               np.std(repeat_stats['n_matches'],axis=0),
               np.mean(repeat_stats['n_repeats'],axis=0),
               np.std(repeat_stats['n_repeats'],axis=0)))

        # Save the winner.
        encoding = NNetPolicy.INPUT_ENC_SIZES[config.genome_config.num_inputs]
        filename = os.path.join(os.getcwd(), NETWORK_DIR, WINNER_FILE % (
            strongweak_string, encoding, config.pop_size, args['n_eval'], p.generation))
        with open(filename, 'wb') as f:
            pickle.dump(winner, f)
            print("\t--------> Saved winner genome, gen %i:  %s" % (p.generation, filename))

        if (iter + 1) % pop_backup_interval == 0 or iter == args['generations'] - 1:
            checkpointer.save_checkpoint(config=config,
                                         population=p.population,
                                         species_set=p.species,
                                         generation=p.generation,
                                         stats=stats)
            print("\t----------> Save population checkpoint: %s" % (filename,))

        visualizer.update(stats)
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    logging.basicConfig(level=logging.INFO)
    run()
