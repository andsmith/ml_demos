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
N_BOOT = 8  # bootstraps for error estimation


class MNISTPairwiseTuner(object):

    def __init__(self, n_cpu=10, pca_dim=DIM, n_samp=1000, no_compute=False):
        self._helper_func = _test_params
        self._dim = pca_dim
        self._no_compute = no_compute  # just load/plot if True
        self._data = MNISTDataPCA(dim=self._dim)
        self._n_samples = n_samp
        self._n_cpu = n_cpu if n_cpu > 0 else cpu_count() - 1
        self._digit_pairs = [(a, b) for a in range(9) for b in range(a+1, 10)]
        logging.info("Anlyzing %i graph types and %i digit pairs." % (len(GRAPH_TYPES), len(self._digit_pairs)))

    def run(self):
        self._results = {}
        for graph_name in GRAPH_TYPES:
            # ,'n-neighbors_mutual','soft_neighbors_additive','soft_neighbors_multiplicative']:
            #if graph_name in ['full', 'epsilon']:
            #    continue
            cache_filename = "%s_%s_n=%i_pca=%i.pkl" % (self._get_prefixes()[1], graph_name, self._n_samples, self._dim)
            self._results[graph_name] = load_cached(
                self._compute, cache_filename, no_compute=self._no_compute, graph_name=graph_name)

    def _get_prefixes(self):
        # logging info, cache filename prefix
        return "spectral clustering", "tuner_pairwise"

    def plot_results(self):
        """
        Plot clustering "accuracies" averaged over all digit pairs for every value of the parameter, with bars indicating +/- 1 sd.

        Plot graph types with k/alpha on the left, epsilon/sigma on the right.
        """
        plot_side = {'n-neighbors_mutual': 0,
                     'n-neighbors': 0,
                     'soft_neighbors_additive': 0,
                     'soft_neighbors_multiplicative': 0,
                     'full': 1,
                     'epsilon': 1}
        axis_labels = {0: 'k/alpha', 1: 'epsilon/sigma'}

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))

        param_vals = {0: None, 1: None}

        def _add_results(ax_ind, results):
            if results is None:
                return
            # START HERE
            param_name = results[0][0]['result'].aux['param_name']
            param_values = [r['param_val'] for r in results[0]]
            # pairs = [r['result'].aux['pair'] for r in results[0]]
            accs = np.array([[r['result'].accuracy for r in result_row] for result_row in results])
            means = np.mean(accs, axis=0)
            sds = np.std(accs, axis=0)
            # print(means.shape, sds.shape)
            param_vals[ax_ind] = param_values

            label = "%s: %s" % (graph_name, param_name)
            self._plot_bands(ax[ax_ind], param_values, means, sds, label)

        for graph_name in self._results:
            g_results = self._results[graph_name]  # list of MNISTResult
            _add_results(plot_side[graph_name], g_results)

        for ax_ind in range(2):
            if param_vals[ax_ind] is not None:
                # set x-scale log
                # ax[ax_ind].set_xscale('log')
                ax[ax_ind].set_xticks(param_vals[ax_ind])
                ax[ax_ind].set_xlabel(axis_labels[ax_ind])
                ax[ax_ind].set_ylabel("Accuracy")
                ax[ax_ind].legend()

        plt.suptitle("Spectral Clustering MNIST Pairwise (pca_dim=%i,\nsamples/digit=%i, avg over 45 digit pairs)" %
                     (self._n_samples, len(self._digit_pairs)))
        plt.legend()
        plt.show()

    def _compute(self, graph_name):
        """
        Look up the paramter name and values, construct the work list, and run the computation.
        """
        param_name = GRAPH_PARAM_NAMES[graph_name]
        param_vals = self._get_param_vals(graph_name, param_name, self._data)
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
                         'param': (param_name, param_vals),
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

    def _get_param_vals(self, graph_name, param_name, data):
        """
        Determine a good set of test values.
        """

        if param_name in ['k', 'alpha']:
            # for the nearest neighbor sim graphs
            # k = np.array([1, 2, 5, 10, 15, 20, 30, 50, 75, 100, 150,200,500])
            k = np.exp(np.linspace(0.0, np.log(500), 22)).astype(int)
            k = np.unique(k)
            print("Testing k values: ", k)
            values = k

        elif param_name in ['epsilon', 'sigma']:
            if True:
                # for epsilon & full graphs (hard/soft thresholding on euclidean distance)
                data = np.vstack([data.get_digit(i) for i in range(10)])
                n_s = 100000
                sample_pairs_a = np.random.choice(data.shape[0], n_s, replace=True)
                sample_pairs_b = np.random.choice(data.shape[0], n_s, replace=True)
                valid = sample_pairs_a != sample_pairs_b
                sample_pairs_a = sample_pairs_a[valid]
                sample_pairs_b = sample_pairs_b[valid]
                distances = np.linalg.norm(data[sample_pairs_a] - data[sample_pairs_b], axis=1)
                # fig, ax = plt.subplots()
                # ax.hist(distances, bins=100)
                # plt.show()
                if param_name == 'epsilon':
                    val_range = np.min(distances)/100, np.percentile(distances, 98)
                else:  # sigma
                    val_range = np.min(distances)/100, np.max(distances)*1.05

                values = np.linspace(val_range[0], val_range[1], 20)  # don't go too low
            else:
                values = np.array([20, 50, 100, 150, 200, 325, 500, 1000, 1500, 2000, 2500])

        else:
            raise ValueError("Unknown param name: %s" % param_name)

        logging.info("Testing values for %s: %s" % (param_name, str(values)))
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

    def __init__(self, n_cpu=6, pca_dim=DIM, n_samp=500, n_boot=N_BOOT, no_compute=False):
        super().__init__(n_cpu, pca_dim, n_samp, no_compute=no_compute)
        self._helper_func = _test_params_full
        self._n_boot = n_boot
        logging.info("Anlyzing %i graph types and %i random samples." % (len(GRAPH_TYPES), self._n_boot))

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
    """
    Subprocess for testing a single set of parameters on a single set of digits.
    :param work: dict with {'graph_name': name of graph type,
                            'param': (param_name, param_values), # to test,
                            'inds': dict(digit -> index_array) for locating images,
                            'data': (x, y) for training/testing,
                            'aux': dict with 'trial' and 'n_trials'  #(which random sample)
                            }
    :return: list of dicts with {'param_val': param, # value tested
                                 'result': MNISTResult  # result of test
                                 }
    """
    graph_name = work['graph_name']
    param_name, params = work['param']
    inds = work['inds']
    x, y = work['data']
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
        result = MNISTResult(10, model, x, y, sample_indices=inds, aux=info)
        print("Graph %s, trial %i/%i, param %s = %s, acc = %f" %
              (graph_name, info['trial']+1, info['n_trials'], param_name, param, result.accuracy))
        results.append({'param_val': param, 'result': result})
    return results


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    
    if False:
        t = MNISTPairwiseTuner(8)
        t.run()
        # t.plot_results()
        del t
    
    if True:
        t = MNISTFullTuner(2)
        t.run()
        t.plot_results()
        del t
