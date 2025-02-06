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


class MNISTPairwiseTuner(object):

    def __init__(self, n_cpu=10, pca_dim=DIM, n_samp=(1000, 500), no_compute=False):
        self._helper_func = _test_params
        self._dim = pca_dim
        self._no_compute = no_compute  # just load/plot if True
        self._data = MNISTData(pca_dim=self._dim)
        self._n_samples, self._n_test = n_samp
        self._n_cpu = n_cpu if n_cpu > 0 else cpu_count() - 1
        self._load_baseline_data()
        self._digit_pairs = [(a, b) for a in range(9) for b in range(a+1, 10)]
        logging.info("Anlyzing %i graph types and %i digit pairs." % (len(GRAPH_TYPES), len(self._digit_pairs)))

    def _load_baseline_data(self):
        pairwise_accuracies = Baselines().data['pairwise']

        def get_acc_stats(which):
            accs = np.array([result[which] for _, result in pairwise_accuracies.items()])
            return {'mean': np.mean(accs), 'sd': np.std(accs)}
        self._baseline_acc = {'train': get_acc_stats('train'),
                              'test': get_acc_stats('test')}
        logging.info("Loaded baseline accuracies for pairwise clustering: %s" % str(self._baseline_acc))

    def run(self):
        self._results = {}
        for graph_name in GRAPH_TYPES:
            cache_filename = "%s_%s_n=%i_pca=%i.pkl" % (self._get_prefixes()[1], graph_name, self._n_samples, self._dim)
            self._results[graph_name] = load_cached(
                self._compute, cache_filename, no_compute=self._no_compute, graph_name=graph_name)

    def _get_prefixes(self):
        # logging info, cache filename prefix
        return "spectral clustering", "tuner_pairwise"

    def _plot_accuracy_curves(self, which='test'):
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
        max_x={0:250, 1:None}

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[1].sharey(ax[0])

        param_vals = {0: None, 1: None}
        param_ranges = {0: None, 1: None}
        n_trials = []

        def _add_results(ax_ind, results):
            if results is None:
                return
            # START HERE
            param_name = results[0][0]['result'].get_info('aux')['param_name']
            param_values = [r['param_val'] for r in results[0]]
            # pairs = [r['result'].aux['pair'] for r in results[0]]
            accs = np.array([[r['result'].accuracy[which] for r in result_row] for result_row in results])
            n_trials.append(accs.shape[0])
            means = np.mean(accs, axis=0)
            sds = np.std(accs, axis=0)
            # print(means.shape, sds.shape)
            param_vals[ax_ind] = param_values if param_vals[ax_ind] is None or \
                np.max(param_values) > np.max(param_vals[ax_ind]) else param_vals[ax_ind]
            param_ranges[ax_ind] = (np.min(param_values), np.max(param_values)) if param_ranges[ax_ind] is None \
                else (np.min((np.min(param_values), param_ranges[ax_ind][0])),
                      np.max((np.max(param_values), param_ranges[ax_ind][1])))

            label = "spectral %s: %s" % (graph_name, param_name)
            self._plot_bands(ax[ax_ind], param_values, means, sds, label)
            return param_values, means
        

        def _add_baseline(ax_ind):
            # show mean/sd of accuracy for kmeans
            # ax[ax_ind].axhline(y=self._baseline_acc[which]['mean'], color='black',
            #                   linestyle='--', label="K-Means" )
            baseline_means = [self._baseline_acc[which]['mean']] * len(param_vals[ax_ind])
            baseline_sd = [self._baseline_acc[which]['sd']] * len(param_vals[ax_ind])
            self._plot_bands(ax[ax_ind], param_vals[ax_ind],
                             baseline_means, baseline_sd, "K-Means avg.", alpha=0, color='black')
        plot_values = {}
        for graph_name in self._results:
            g_results = self._results[graph_name]  # list of MNISTResult
            p_vals, mean_acc = _add_results(plot_side[graph_name], g_results)
            plot_values[graph_name] = {'param_vals': p_vals, 'mean_acc': mean_acc, 'plot_side': plot_side[graph_name]}

        for ax_ind in range(2):
            _add_baseline(ax_ind)
            if param_vals[ax_ind] is not None:
                # set x-scale log
                #ax[ax_ind].set_xscale('log')
                y_lim_0 = max(0, ax[ax_ind].get_ylim()[0])
                y_lim_1 = min(1, ax[ax_ind].get_ylim()[1])
                ax[ax_ind].set_ylim([y_lim_0, y_lim_1])
                # x_0,_ = ax[ax_ind].get_xlim()
                ax[ax_ind].set_xlim([0, max_x[ax_ind]])
                ax[ax_ind].set_ylabel("Accuracy +/- 1 sd.")
                ax[ax_ind].legend()
        if len(np.unique(n_trials)) > 1:
            logging.warning("Different number of trials for different graph types: %s" % str(n_trials))
        n_trials = n_trials[0]
        plt.legend()
        return fig, ax, plot_values

    def plot_results(self, which='test'):
        self._plot_accuracy_curves(which)
        title = "Spectral clustering - MNIST pairwise, %s data\n(pca_dim=%i, samples/digit=%i, avgs over %i trials)" %\
            (which, self._dim, self._n_samples, len(self._digit_pairs))
        plt.suptitle(title)

    def _compute(self, graph_name):
        """
        Look up the paramter name and values, construct the work list, and run the computation.
        """
        param_name = GRAPH_PARAM_NAMES[graph_name]
        param_vals = self._get_param_vals(graph_name, param_name, self._data)
        work = []
        for pair in self._digit_pairs:

            data = self._data.get_sample(digits=pair,
                                         n_train=self._n_samples,
                                         n_test=self._n_test)

            work.append({'graph_name': graph_name,
                         'aux': {'pair': pair},
                         'data': data,
                         'param': (param_name, param_vals),
                         'k': 2, }
                        )
        logging.info("About to run pairwise tuning with %i tasks for graph type:  %s." % (len(work), graph_name))
        if self._n_cpu == 1:
            logging.info("Running %s in single process." % self._get_prefixes()[0])
            results = [self._helper_func(w) for w in work]
        else:
            logging.info("Running %s in %i processes." % (self._get_prefixes()[0], self._n_cpu))
            with Pool(self._n_cpu) as pool:
                results = pool.map(self._helper_func, work)
        return results

    def _plot_bands(self, ax, param_values, means, sds, label, alpha=.2, color=None):
        c_args = {'color': color} if color is not None else {}
        ax.plot(param_values, means, label=label, **c_args)
        ax.fill_between(param_values,
                        np.array(means) - np.array(sds),
                        np.array(means) + np.array(sds),
                        alpha=alpha, **c_args)

    def _get_param_vals(self, graph_name, param_name, data):
        """
        Determine a good set of test values.
        """

        if param_name in ['k', 'alpha']:
            # for the nearest neighbor sim graphs
            # k = np.array([1, 2, 5, 10, 15, 20, 30, 50, 75, 100, 150,200,500])
            k = np.exp(np.linspace(0.0, np.log(500), 30)).astype(int)
            k = np.unique(k)
            print("Testing k values: ", k)
            values = k

        elif param_name in ['epsilon', 'sigma']:
            if True:
                # for epsilon & full graphs (hard/soft thresholding on euclidean distance)
                x = np.vstack([data.train[i] for i in range(10)])
                n_s = 100000
                sample_pairs_a = np.random.choice(x.shape[0], n_s, replace=True)
                sample_pairs_b = np.random.choice(x.shape[0], n_s, replace=True)
                valid = sample_pairs_a != sample_pairs_b
                sample_pairs_a = sample_pairs_a[valid]
                sample_pairs_b = sample_pairs_b[valid]
                distances = np.linalg.norm(x[sample_pairs_a] - x[sample_pairs_b], axis=1)
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
    samp = work['data']
    x, y = samp.get_data('train')
    k = work['k']
    results = []
    for param in params:
        sim_graph = GRAPH_TYPES[graph_name](x, **{param_name: param})
        model = SpectralAlgorithm(sim_graph)
        model.fit(k, k)
        info = work['aux'].copy()
        info['param_val'] = param
        info['param_name'] = param_name
        result = MNISTResult(k, model, samp)
        result.set_info('aux', info)
        print("Graph %s, pair %s, param %s = %s, acc = %s" %
              (graph_name, info['pair'], param_name, param, result.accuracy))
        results.append({'param_val': param, 'result': result})
    return results


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


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    if True:
        t = MNISTPairwiseTuner(6)
        t.run()
        t.plot_results()
        del t

    if True:
        t = MNISTFullTuner(5)
        t.run()
        t.plot_results()
        del t

    plt.show()
