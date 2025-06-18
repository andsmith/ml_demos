from policies import Policy
from game_base import Mark
import numpy as np
from neat.nn import FeedForwardNetwork as NNet  # Assuming NNet is a feedforward neural network
import neat
import neat.reporting
import time
from neat.reporting import iterkeys


class ParallelEvaluatorWStats(neat.ParallelEvaluator):
    def __init__(self, n_workers, eval_function, timeout=None):
        super().__init__(n_workers, eval_function, timeout)
        self.reset_stats()

    def reset_stats(self):
        self.stats = {'n_matches': [],
                      'n_repeats': []}

    def evaluate(self, genomes, config):
        jobs = []
        for ignored_genome_id, genome in genomes:
            jobs.append(self.pool.apply_async(self.eval_function, (genome, config)))
        # assign the fitness back to each genome
        for job, (ignored_genome_id, genome) in zip(jobs, genomes):
            genome.fitness, repeat_stats = job.get(timeout=self.timeout)
            self.stats['n_matches'].append(repeat_stats['n_matches'])
            self.stats['n_repeats'].append(repeat_stats['n_repeats'])



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



class NNetPolicy(Policy):

    INPUT_ENC_SIZES = {9: 'enc',
                       18: 'enc+free',  # 9 + 9 free actions
                       27: 'one-hot'}  # 9 + 18 free actions

    def __init__(self, nnet):
        self._net = nnet
        input_size = len(nnet.input_nodes)
        self.encoding = self.INPUT_ENC_SIZES[input_size]
        self.player = Mark.X
        if self.encoding not in ['enc', 'enc+free', 'one-hot']:
            raise ValueError("encoding must be one of 'enc', 'enc+free', or 'one-hot'")
        
    def _net_out_to_action(self, state,output):
        open_actions = state.get_actions(flat_inds=True)
        valid_outputs = output[open_actions]
        bi = np.argmax(valid_outputs)
        best_action_flat = open_actions[bi]  # get the index of the best action in the flat array
        best_action = (best_action_flat // 3, best_action_flat % 3)  # convert to (i, j) tuple
        return [(best_action, 1.0)]  # return the best action with probability 1.0


    def recommend_action(self, state):
        """
        Given a state, return the action disrtibution (deterministic for nnet, highest output for free actions).
        :param state: The current state of the game.
        :returns: a list of (action, prob) tuples, where prob is the probability of taking the action.
        """
        inputs = state.to_nnet_input(method=self.encoding)  # get the input for the neural network
        output = np.array(self._net.activate(inputs))
        return self._net_out_to_action(state,output)
