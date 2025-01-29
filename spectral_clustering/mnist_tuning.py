"""
Sweep over parameter values for the spectral clustering variants.

In "Pairwise Mode", for each parameter value:
    * Each of the 45 digit pairs is clustered with K=2, cluster/class labels are assigned to maximize accuracy.
    * Mean/sd of accuracy is computed over all pairs.
    * Results are a plot of the accuracy +/- 1 sd curve as the parameter varies.
    * Add K-Means and Fisher LDA for comparison.

In "Full Mode", for each parameter value:
    * All digits are clustered with K=10, cluster/class labels are assigned to maximize accuracy.
    * Accuracy computed as the mean (and sd) of class label correctness.
    * Results are a plot of the accuracy +/- 1 sd curve as the parameter varies.
    * Add K-Means for comparison.

"""
import numpy as np
import matplotlib.pyplot as plt
import logging
from mnist_common import GRAPH_TYPES, GRAPH_PARAM_NAMES, MNISTResult
from clustering import SpectralAlgorithm, KMeansAlgorithm
from multiprocessing import Pool, cpu_count
from pprint import pprint
from mnist_data import MNISTDataPCA
from mpl_plots import project_binary_clustering, plot_binary_clustering
import pickle
import os
from fisher_lda import FisherLDA
from scipy.spatial.distance import pdist
from util import load_cached
# Common params for full and pairwise experiments
DIM = 30
N_SAMP = 1000
N_BOOT = 50  # bootstraps for error estimation
N_CPU = 12
n_vals = 50


class MNISTPairwiseTuner(object):

    def __init__(self, n_param_vals=n_vals, n_cpu=N_CPU, pca_dim=DIM, n_samp=N_SAMP):
        self._helper_func = _test_params
        self._dim = pca_dim
        self._data = MNISTDataPCA(dim=self._dim)
        self._n_samples = n_samp
        self._n_param_vals = n_param_vals
        self._n_cpu = n_cpu if n_cpu > 0 else cpu_count() - 1
        self._digit_pairs = [(a, b) for a in range(9) for b in range(a+1, 10)]

    def run(self):
        # compute/load
        logging.info("Anlyzing %i graph types and %i digit pairs." % (len(GRAPH_TYPES), len(self._digit_pairs)))
        self._results = {}
        for graph_name in  GRAPH_TYPES:
            cache_filename = "%s_%s_n=%i.pkl" % (self._get_prefixes()[1], graph_name, self._n_samples)
            self._results[graph_name] = load_cached(lambda: self._compute(graph_name), cache_filename, no_compute=False)

    def _get_prefixes(self):
        # logging info, cache filename prefix
        return "spectral clustering", "tuner_pairwise"

    def plot_results(self):
        """
        plot the mean (over all pairs) accuracy curve and +/- 1 sd deviation band for each graph type.
        Add a legend.

        self._results should now be a dict with one entry per graph_name, each
           value is a list of lists of MNISTResult objects:
              for all pairs P:
                 for all parameter values V:
                    MNISTResult(results on P and V)

            (unfortunately, a bit inside-out for plotting curves over parameter values)
        """
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))

        def _add_results(results):

            # START HERE

            param_name = results[0][0]['result'].aux['param_name']
            param_values = [r['param_val'] for r in results[0]]
            pairs = [r['result'].aux['pair'] for r in results[0]]
            accs = np.array([[r['result'].accuracy for r in result_row] for result_row in results])
            means = np.mean(accs, axis=0)
            sds = np.std(accs, axis=0)
            # print(means.shape, sds.shape)

            label = "%s: %s" % (graph_name, param_name)
            self._plot_bands(ax, param_values, means, sds, label)

        for graph_name in self._results:
            g_results = self._results[graph_name]  # list of MNISTResult
            _add_results(g_results)
        ax.set_ylabel("Accuracy")
        plt.legend()
        plt.show()

    def _compute(self, graph_name):
        """
        Look up the paramter name and values, construct the work list, and run the computation.
        """
        param_name = GRAPH_PARAM_NAMES[graph_name]
        param_range = self._get_param_range(graph_name, param_name, self._n_param_vals, self._data)
        work = []
        for pair in self._digit_pairs:
            # sample randomly from each digit, each time.
            inds0 = self._data.get_digit_sample_inds(pair[0], self._n_samples)
            inds1 = self._data.get_digit_sample_inds(pair[1], self._n_samples)
            inds = {0: inds0, 1: inds1}
            x = np.vstack((self._data.get_digit(pair[0])[inds0],
                           self._data.get_digit(pair[1])[inds1]))
            y = np.hstack((np.zeros(inds0.size),
                           np.ones(inds1.size)))
            work.append({'graph_name': graph_name,
                         'aux': {'pair': pair},  # for debug print
                         'param': (param_name, param_range),
                         'k': 2,
                         'inds': inds,  # for locating image
                         'data': (x, y), })
        logging.info("About to run pairwise tuning with %i tasks for graph type:  %s." % (len(work), graph_name))
        if self._n_cpu == 1:
            logging.info("Running %s in single process." % self._get_prefixes()[0])
            results = [self._helper_func(w) for w in work]
        else:
            logging.info("Running %s in %i processes." % (self._get_prefixes()[0], self._n_cpu))
            with Pool(self._n_cpu) as pool:
                results = pool.map(self._helper_func, work)
        return results

    def _plot_bands(self, ax, param_values, means, sds, label):
        ax.plot(param_values, means, label=label)
        ax.fill_between(param_values,
                        np.array(means) - np.array(sds),
                        np.array(means) + np.array(sds),
                        alpha=0.2)

    def _get_param_range(self, graph_name, param_name, n_vals, data):
        """
        Determine a good set of test values.
        """

        if param_name == 'k':
            # for the nearest neighbor sim graphs
            values = np.arange(1, n_vals).astype(int)

        elif param_name in ['epsilon', 'sigma']:
            # for epsilon & full graphs (hard/soft thresholding on euclidean distance)
            data = np.vstack([data.get_digit(i) for i in range(10)])
            n_s = 10000
            sample_pairs_a = np.random.choice(data.shape[0], n_s, replace=True)
            sample_pairs_b = np.random.choice(data.shape[0], n_s, replace=True)
            valid = sample_pairs_a != sample_pairs_b
            sample_pairs_a = sample_pairs_a[valid]
            sample_pairs_b = sample_pairs_b[valid]
            distances = np.linalg.norm(data[sample_pairs_a] - data[sample_pairs_b], axis=1)
            #fig, ax = plt.subplots()
            #ax.hist(distances, bins=100)
            #plt.show()
            if param_name == 'epsilon':
                val_range = np.min(distances), np.percentile(distances, (98))
            else:  # sigma
                print("MIN DISTANCE", np.min(distances))
                val_range = np.min(distances)/100, np.percentile(distances, (20))

            values = np.linspace(val_range[0], val_range[1], n_vals)  # don't go too low

        elif param_name == 'alpha':
            # for the soft nearest neighbors graphs
            values = np.linspace(0.1, 50, n_vals)
        else:
            raise ValueError("Unknown param name: %s" % param_name)

        logging.info("%s range: %f to %f  (%i values)" % (param_name, values[0], values[-1], values.size))
        return values


def _test_params(work):
    """
    Subprocess for testing a single set of parameters on a single pair of digits.
    :param work: dict with {'graph_name': name of graph type,
                            'param': (param_name, param_values), # to test,
                            'k': number of clusters,
                            'inds': (inds0, inds1) for locating image,
                            'data': (x, y) for training/testing}
    :return: list of MNISTResult with the parameter value in the 'aux' field.
    """

    graph_name = work['graph_name']
    param_name, params = work['param']
    inds = work['inds']
    x, y = work['data']
    k = work['k']
    results = []
    for param in params:
        sim_graph = GRAPH_TYPES[graph_name](x, **{param_name: param})
        model = SpectralAlgorithm(sim_graph)
        model.fit(k, k)
        info = work['aux'].copy()
        info['param_val'] = param
        info['param_name'] = param_name
        result = MNISTResult(k, model, x, y, sample_indices=inds, aux=info)
        print("Graph %s, pair %s, param %s = %s, acc = %f" %
              (graph_name, info['pair'], param_name, param, result.accuracy))
        results.append({'param_val': param, 'result': result})
    return results


class MNISTFullTuner(MNISTPairwiseTuner):
    """
    Tuning params for clustering the 10 classes at ones.
    Parallelism is over bootstrap samples.
    """

    def __init__(self, n_param_vals=n_vals, n_cpu=N_CPU, pca_dim=DIM, n_samp=N_SAMP, n_boot=N_BOOT):
        super().__init__(n_param_vals, n_cpu, pca_dim, n_samp)
        self._helper_func = _test_params_full
        self._n_boot = n_boot

    def run(self):
        # compute/load
        logging.info("Anlyzing %i graph types and %i random samples." % (len(GRAPH_TYPES), self._n_boot))
        self._results = {}
        for graph_name in ['n-nearest']:  # GRAPH_TYPES:
            cache_filename = "%s_%s_n=%i.pkl" % (self._get_prefixes()[1], graph_name, self._n_samples)
            self._results[graph_name] = load_cached(lambda: self._compute(graph_name), cache_filename)

    def _get_prefixes(self):
        # logging info, cache filename prefix
        return "spectral clustering", "tuner_full"

    def _compute(self, graph_name):
        """
        Look up the paramter name and values, construct the work list, and run the computation.
        """
        param_name = GRAPH_PARAM_NAMES[graph_name]
        param_range = self._get_param_range(graph_name, param_name, self._n_param_vals, self._data)
        work = []
        for i in range(self._n_boot):
            # sample randomly from each digit, each time.
            inds = {d: self._data.get_digit_sample_inds(d, self._n_samples) for d in range(10)}

            x = np.vstack([self._data.get_digit(d)[inds[d]] for d in range(10)])
            y = np.hstack([d*np.ones(inds[d].size) for d in range(10)])

            work.append({'graph_name': graph_name,
                         'inds': inds,
                         'data': (x, y),
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


def _test_params_full(work):
    raise NotImplementedError("Implement me!")


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    # test_data()
    t = MNISTPairwiseTuner()
    t.run()
    t.plot_results()
