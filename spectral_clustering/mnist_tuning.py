"""
Sweep over parameter values for the spectral clustering variants.

In "Pairwise Mode", for each parameter value:
    * Each of the 45 digit pairs is clustered with K=2, cluster/class labels are assigned to maximize accuracy.
    * Mean/sd of accuracy is computed over all pairs.
    * Results are a plot of the accuracy +/- 1 sd curve as the parameter varies.
    * Add K-Means and Fisher LDA for comparison.

In "Full Mode", for each parameter value:
    * All digits are clustered with K=10, cluster/class labels are assigned to maximize accuracy.
    * Accuracy computed as the mean (and sd) of the one-vs-rest classification accuracy.
    * Results are a plot of the accuracy +/- 1 sd curve as the parameter varies.
    * Add K-Means for comparison.

"""
import numpy as np
import matplotlib.pyplot as plt
import logging
from similarity import GRAPH_TYPES
from clustering import SpectralAlgorithm, KMeansAlgorithm, ClusterClassifier
from multiprocessing import Pool, cpu_count
from pprint import pprint
from mnist_data import MNISTData
from mpl_plots import project_binary_clustering, plot_binary_clustering, show_digit_cluster_collage
import pickle
import os
from fisher_lda import FisherLDA
# Expand these for easier comparison:



class MNISTTuner(object):


    def __init__(self, data, n_KM_trials=30, n_param_tests=30, n_cpu=1):
        self._data = data
        self._n_KM_trials = n_KM_trials
        self._init_tests(n_cpu, n_param_tests)
        logging.info("Running with {} digit pairs.".format(len(self._digit_pairs)))

    def _init_tests(self, n_cpu, n_param_tests):
        self._k_means_results = None
        self._spectral_results = None
        self._n_cpu = n_cpu if n_cpu > 0 else cpu_count() - 1
        self._digit_pairs = [(a, b) for a in range(9) for b in range(a+1, 10)]

        self._param_ranges = {graph_name: self._get_param_range(graph_name,
                                                                self.GRAPH_PARAM_NAMES[graph_name],
                                                                n_param_tests,
                                                                self._data)
                              for graph_name in self.GRAPH_PARAM_NAMES}

    def _get_train_test_data(self, pair):
        data_a_train, data_a_test = self._data.get_digit(pair[0])
        data_b_train, data_b_test = self._data.get_digit(pair[1])
        x_train = np.vstack((data_a_train, data_b_train))
        x_test = np.vstack((data_a_test, data_b_test))
        y_train = np.concatenate((np.zeros(len(data_a_train)), np.ones(len(data_b_train))))
        y_test = np.concatenate((np.zeros(len(data_a_test)), np.ones(len(data_b_test))))

        img_a_train, img_a_test = self._data.get_images(pair[0])
        img_b_train, img_b_test = self._data.get_images(pair[1])
        img_train = np.vstack((img_a_train, img_b_train))
        img_test = np.vstack((img_a_test, img_b_test))

        return {'x_train': x_train,
                'y_train': y_train,
                'x_test': x_test,
                'y_test': y_test,
                'img_train': img_train,
                'img_test': img_test,
                'pair': pair}

 


    def _stats_from_results(self, results):
        train_accs = [r['acc_train'] for r in results]
        test_accs = [r['acc_test'] for r in results]

        return {'train_mean': np.mean(train_accs),
                'train_sd': np.std(train_accs),
                'test_mean': np.mean(test_accs),
                'test_sd': np.std(test_accs)}

    def _get_spectral_results(self):
        """
        :param graph_name: name of the graph type to use
        :param n_param_vals: number of parameter values to test
        """

        work = []
        logging.info("Creating work for anlyzing %i graph types and %i digit pairs." %
                     (len(self.GRAPH_TYPES), len(self._digit_pairs)))

        for graph_name in self.GRAPH_TYPES:
            param_name = self.GRAPH_PARAM_NAMES[graph_name]
            for pair in self._digit_pairs:
                data = self._get_train_test_data(pair)
                param_range = self._param_ranges[graph_name]
                work.append({'pair': pair,
                             'data': data,
                             'graph_name': graph_name,
                             'param_name': param_name,
                             'param_vals': param_range})

        logging.info("Done creating work for analyzing %i graph types:" % len(self.GRAPH_TYPES))
        logging.info("\t%i digit pairs" % len(self._digit_pairs))
        logging.info("\t%i tasks" % (len(work)))
        logging.info("\t%i parameters values per task" % len(param_range))
        logging.info("\t%i training samples per class" % self._data.n_train)
        if self._n_cpu == 1:
            output = [_test_params(w) for w in work]
        else:
            with Pool(self._n_cpu) as pool:
                output = pool.map(_test_params, work)
        # flatten list of results (now a list of lists, one per graph type, per pair)
        output = [r for sublist in output for r in sublist]
        # reassemble: for each graph type and param value, get stats
        results = {}
        for graph_name in self.GRAPH_TYPES:
            g_out = [r for r in output if r['graph_name'] == graph_name]
            param_vals = sorted(set([r['param_val'] for r in g_out]))
            results[graph_name] = {}
            for param_val in param_vals:
                p_out = [r for r in g_out if r['param_val'] == param_val]
                results[graph_name][param_val] = self._stats_from_results(p_out)

        return results



    def run(self):
        self._k_means_results = self._get_kmeans_results()
        self._k_means_stats = self._stats_from_results(self._k_means_results)
        # _, ax = plt.subplots(1, 1)
        # _, ax2 = plt.subplots(1, 1)

        # self.plot_pairwise_clusterings(ax, self._k_means_results, 'train')
        # self.plot_pairwise_clusterings(ax2, self._fisher_results, 'train')

        # self._spectral_results = self._get_spectral_results()
        # self.plot_km_digit_confusion()

        # self._plot_best_and_worst_pairs(self._fisher_results, n=10)
        self._plot_best_and_worst_pairs(self._k_means_results, n=10)

        logging.info("KMeans results:")
        pprint(self._k_means_stats)
        # fig, ax = plt.subplots()
        # self._plot_spectral_result(ax, 'n-neighbors', 'train')
        # ax.set_title("K-Means cluster separation")
        # ax2.set_title("Fisher LDA class separation")
        plt.show()

    def _plot_bands(self, ax, param_values, means, sds, label):
        ax.plot(param_values, means, label=label)
        ax.fill_between(param_values,
                        np.array(means) - np.array(sds),
                        np.array(means) + np.array(sds),
                        alpha=0.2)

    def _plot_spectral_result(self, ax, graph_name, test_train='test'):
        """
        Plot the results of the spectral clustering for a single graph type.
        Add the K-Means baseline for comparison in red.
        :param graph_name: name of the graph type to plot
        :return:
        """

        param_vals = sorted(self._spectral_results[graph_name].keys())
        means = [self._spectral_results[graph_name][param_val][test_train + '_mean'] for param_val in param_vals]
        sds = [self._spectral_results[graph_name][param_val][test_train + '_sd'] for param_val in param_vals]
        self._plot_bands(ax, param_vals, means, sds, graph_name)
        self._plot_kmean_baseline(ax, test_train)

    def _get_param_range(self, graph_name, param_name, n_vals, data):
        """
        Determine a good set of test values.
        """

        if param_name == 'k':
            return np.arange(1, n_vals).astype(int)
        elif param_name == 'sigma':
            return np.linspace(0.1, 5, n_vals)
        elif param_name == 'epsilon':
            return np.linspace(0.01, 1, n_vals)
        elif param_name == 'alpha':
            return np.linspace(0.1, 5, n_vals)


def _test_params(work):
    """
    Subprocess for testing a single set of parameters on a single pair of digits.
    :param work: dict with {'graph_name': key of self.GRAPH_TYPES,
                    'param_name': name of the parameter to vary,
                    'param_vals': list of values to test,
                    'pair': (a, b),
                    'data': dict with x_train, x_test, y_train, y_test}

    :return: list of dicts with {'acc_train': train accuracy,
                                 'acc_test': test accuracy,
                                 'pair': (a, b),
                                 'graph_name': (same as input),
                                 'param_name': (same as input),
                                 'param_val': value of the parameter}
            for every parameter value tested.
    """

    data = work['data']
    x_train = data['x_train']
    y_train = data['y_train']
    x_test = data['x_test']
    y_test = data['y_test']
    graph_name = work['graph_name']
    param_name = work['param_name']
    param_vals = work['param_vals']
    graph_type = MNISTTuner.GRAPH_TYPES[graph_name]
    results = []
    for param_val in param_vals:
        graph = graph_type(x_train, **{param_name: param_val})
        clust = SpectralAlgorithm(graph)
        clust.fit(n_clusters=2, n_features=2)
        classif = ClusterClassifier(clust, x_train, y_train)

        # determine best train accuracy
        train_out = classif.predict(x_train)
        test_out = classif.predict(x_test)

        train_acc = np.mean(train_out == y_train)
        test_acc = np.mean(test_out == y_test)

        results.append({'acc_train': train_acc,
                        'acc_test': test_acc,
                        'pair': work['pair'],
                        'graph_name': graph_name,
                        'param_name': param_name,
                        'param_val': param_val})

        print("Tested %s on digits %s with %s=%s:  (Train acc:  %.4f)" %
              (graph_name, work['pair'], param_name, param_val, train_acc))

    return results


def tune_all():
    data = MNISTData(n_test=1000, n_train=2500, dim=30)
    tuner = MNISTTuner(data, n_cpu=10)
    tuner.run()


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    # test_data()
    tune_all()
