"""
Modified copy of neat.checkpoint.py:  Extended to save stats & generation number (should now be totall resumable,
show plots same as if it hadn't been interrupted then resumed, etc).

Uses `pickle` to save and restore populations (and other aspects of the simulation state)."""
from __future__ import print_function

import gzip
import random
import time

try:
    import cPickle as pickle # pylint: disable=import-error
except ImportError:
    import pickle # pylint: disable=import-error

from neat.population import Population
from neat_util import StdOutReporterWGenomSizesPlus

import neat

class CheckpointerWithStats(neat.Checkpointer):

    def save_checkpoint(self, config, population, species_set, generation, stats):
        """ Save the current simulation state. """
        filename = '{0}{1}'.format(self.filename_prefix,generation)
        print("Saving checkpoint to {0}".format(filename))
        with gzip.open(filename, 'wb', compresslevel=5) as f:
            data = (generation, config, population, species_set, random.getstate(), stats)
            pickle.dump(data, f)

    @staticmethod
    def restore_checkpoint_old(filename):
        # no stats, old behavior
        return super(CheckpointerWithStats, CheckpointerWithStats).restore_checkpoint(filename)

    @staticmethod
    def restore_checkpoint_new(filename):
        with gzip.open(filename, 'rb') as f:
            generation, config, population, species_set, rndstate, stats = pickle.load(f)
            random.setstate(rndstate)
            restored_population = Population(config, (population, species_set, generation))
            restored_population.generation = generation
            return restored_population, stats
        
    @staticmethod
    def restore_checkpoint(filename):
        try:
            return CheckpointerWithStats.restore_checkpoint_new(filename)
        except ValueError as e:
            print(f"Failed to restore new checkpoint format: {e}. Trying old format...")
            return CheckpointerWithStats.restore_checkpoint_old(filename)    
        