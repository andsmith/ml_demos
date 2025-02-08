"""
Spectral clustering on all 10 digits at once:
  * Do clusters correspond to digits?

To investigate, for each spectral algorithm variant:
  1. cluster the data using K=10
  2. Assign each cluster to a digit using the Hungarian algorithm.
  3. Compute the confusion matrix and a 2-d embedding to show the clusters.

Plot results in 3 figures:
  * Confusion matrix with (mean) accuracy as a title, each algorithm (and K-means) in a subplot.
  * Box-plots of mean accuracies over N trials (random subset of data)
  * The cluster embeddings, each in separate subplots.

"""

import numpy as np
import matplotlib.pyplot as plt
import logging
from mnist_common import GRAPH_TYPES, GRAPH_PARAM_NAMES, MNISTResult, Baselines
from clustering import SpectralAlgorithm, KMeansAlgorithm
from multiprocessing import Pool, cpu_count
from pprint import pprint
from mnist_data import MNISTData
from mpl_plots import project_binary_clustering, plot_binary_clustering
from mnist_plots import show_digit_cluster_collage_full, plot_extreme_pairs, plot_pairwise_clusterings
import pickle
import os
from fisher_lda import FisherLDA
from scipy.spatial.distance import pdist
from util import load_cached
# Common params for full and pairwise experiments
DIM = 30
N_BOOT = 5  # bootstraps for error estimation

from mnist_pairwise import MNISTPairwiseTuner


class MNISTFullTuner(MNISTPairwiseTuner):
    """
    Tuning params for clustering the 10 classes at ones.
    Parallelism is over bootstrap samples.
    """

    def __init__(self, n_cpu=2, pca_dim=DIM, n_samp=(500, 500), n_boot=N_BOOT, no_compute=False):
        super().__init__(n_cpu, pca_dim, n_samp, no_compute=no_compute)
        self._helper_func = _test_params_full
        self._n_boot = n_boot
        logging.info("Anlyzing %i graph types and %i random samples." % (len(GRAPH_TYPES), self._n_boot))

    def _load_baseline_data(self):
        acc = Baselines().data['full']
        self._baseline_acc = {'test': {'mean': acc['test'], 'sd': 0.0},
                              'train': {'mean': acc['train'], 'sd': 0.0}}

        logging.info("Loaded baseline accuracies for full clustering: %s" % str(self._baseline_acc))

    def _get_prefixes(self):
        # logging info, cache filename prefix
        return "spectral clustering", "tuner_full"

    def _compute(self, graph_name):
        """
        Look up the paramter name and values, construct the work list, and run the computation.
        """
        param_name = GRAPH_PARAM_NAMES[graph_name]
        param_range = self._get_param_vals(graph_name, param_name, self._data)
        work = []
        for i in range(self._n_boot):
            sample = self._data.get_sample(digits=None, n_train=self._n_samples, n_test=self._n_test)
            work.append({'graph_name': graph_name,
                         'data': sample,
                         'param': (param_name, param_range),
                         'aux': {'trial': i, 'n_trials': self._n_boot}})
        logging.info("About to run full tuning with %i tasks for graph type:  %s." % (len(work), graph_name))

        if self._n_cpu == 1:
            logging.info("Running %s in single process." % self._get_prefixes()[0])
            results = [self._helper_func(w) for w in work]
        else:
            logging.info("Running %s in %i processes." % (self._get_prefixes()[0], self._n_cpu))
            with Pool(self._n_cpu) as pool:
                results = pool.map(self._helper_func, work)
        return results

    def plot_results(self, which='test'):
        self._plot_accuracy_curves(which)
        key0 = [k for k in self._results][0]
        n_boot = len(self._results[key0]    )
        title = "Spectral clustering - MNIST 10-class, %s data\n(pca_dim=%i, samples/digit=%i, avgs over %i trials)" %\
            (which, self._dim, self._n_samples, n_boot)
        plt.suptitle(title)


def _test_params_full(work):
    """
    Subprocess for testing a single set of parameters on a single set of digits.
    :param work: dict with {'graph_name': name of graph type,
                            'param': (param_name, param_values), # to test,
                            'data':MNISTSample object,
                            'aux': dict with 'trial' and 'n_trials'  #(which random sample)
                            }
    :return: list of dicts with {'param_val': param, # value tested
                                 'result': MNISTResult  # result of test
                                 }
    """
    graph_name = work['graph_name']
    param_name, params = work['param']
    data = work['data']
    x, _ = data.get_data('train')
    results = []
    print("Starting full test with %s, %s, %i samples (trial %i of %i)." %
          (graph_name, param_name, x.shape[0], 1+work['aux']['trial'], work['aux']['n_trials']))
    for param in params:
        sim_graph = GRAPH_TYPES[graph_name](x, **{param_name: param})
        model = SpectralAlgorithm(sim_graph)
        model.fit(10, 10)
        info = work['aux'].copy()
        info['param_val'] = param
        info['param_name'] = param_name
        result = MNISTResult(10, model, data)
        result.set_info('aux', info)
        print("\tgraph %s, trial %i/%i, param %s = %s, acc = %s" %
              (graph_name, info['trial']+1, info['n_trials'], param_name, param, result.accuracy))
        results.append({'param_val': param, 'result': result})
    return results


if __name__ == '__main__':
        

    t = MNISTFullTuner(5)
    t.run()
    t.plot_results()
    plt.show()
