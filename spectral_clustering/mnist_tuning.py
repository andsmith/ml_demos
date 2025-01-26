"""
What are the effects of the different similarity graphs & their parameters on the
spectrum and clustering results?

    method:
      1. Reduce dim to 30 with PCA.
      2. For each clustering algorithm variant, iterate over many values of its parameter:
      3.    For all 45 pairs of dissimilar digits:
      4.       Cluster with K=2, determine cluster labels that yield best training accuracy.
      5     Compute mean, sd of each parameter value (and k-means).
      6. For the N pairs, cluster 5 times with K-Means, determine mean & sd of test/train accuracy.
      7. Plot the accuracy +/- 1 sd curve as the parameter varies. 
      8. Plot the best result for each algorithm (and k-means on each) on a single plot.
"""
import numpy as np
import matplotlib.pyplot as plt
import logging
from similarity import FullSimGraph, NNSimGraph, SoftNNSimGraph, EpsilonSimGraph
from clustering import SpectralAlgorithm, KMeansAlgorithm, ClusterClassifier
from multiprocessing import Pool, cpu_count
from pprint import pprint
from mnist_data import MNISTData
from mpl_plots import project_binary_clustering, plot_binary_clustering, show_digit_cluster_collage
import pickle
import os
from fisher_lda import FisherLDA
# Expand these for easier comparison:


class SoftNNSimGraphAdditive(SoftNNSimGraph):
    def __init__(self, points, alpha):
        super().__init__(points, alpha, additive=True)


class SoftNNSimGraphMultiplicative(SoftNNSimGraph):
    def __init__(self, points, alpha):
        super().__init__(points, alpha, additive=False)


class NNSimGraphMutual(NNSimGraph):
    def __init__(self, points, k):
        super().__init__(points, k, mutual=True)


class NNSimGraphNonMutual(NNSimGraph):
    def __init__(self, points, k):
        super().__init__(points, k, mutual=False)


class MNISTTuner(object):

    GRAPH_TYPES = {  # 'full': FullSimGraph,
        'n-neighbors': NNSimGraphNonMutual,
        # 'n-mutual-neighbors': NNSimGraphMutual,
        # 'soft_neighbors_additive': SoftNNSimGraphAdditive,
        # 'soft_neighbors_multiplicative': SoftNNSimGraphMultiplicative,
        # 'epsilon': EpsilonSimGraph
    }

    GRAPH_PARAM_NAMES = {'full': 'sigma',
                         'n-neighbors': 'k',
                         'n-mutual-neighbors': 'k',
                         'soft_neighbors_additive': 'alpha',
                         'soft_neighbors_multiplicative': 'alpha',
                         'epsilon': 'epsilon'}

    KM_RESULTS_CACHE_FILE = "km_results.pkl"
    SPECTRAL_RESULTS_CACHE_FILE = "spectral_results.pkl"
    FISHER_CACHE_FILE = "fisher_results.pkl"

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

    def _get_kmeans_results(self):
        if not os.path.exists(self.KM_RESULTS_CACHE_FILE):
            work = []
            logging.info("Creating work for K-Means results...")
            for pair in self._digit_pairs:
                work.append({'pair': pair,
                            'data': self._get_train_test_data(pair),
                             'n_trials': self._n_KM_trials})
            logging.info("Done creating work for K-Means results, %i tasks." % len(work))

            if self._n_cpu == 1:
                logging.info("Running K-Means results in single process.")
                results = [_test_kmeans(w) for w in work]
            else:
                logging.info("Running K-Means results in %i processes." % self._n_cpu)
                with Pool(self._n_cpu) as pool:
                    results = pool.map(_test_kmeans, work)

            with open(self.KM_RESULTS_CACHE_FILE, 'wb') as f:
                logging.info("Saving K-Means results to %s" % self.KM_RESULTS_CACHE_FILE)
                pickle.dump(results, f)
        else:

            with open(self.KM_RESULTS_CACHE_FILE, 'rb') as f:
                logging.info("Loading K-Means results from %s" % self.KM_RESULTS_CACHE_FILE)
                results = pickle.load(f)

        return results

    def plot_km_digit_confusion(self):
        test_img = np.zeros((10, 10), dtype=np.float32)
        train_img = np.zeros((10, 10), dtype=np.float32)
        for result in self._k_means_results:
            train_img[int(result['pair'][0]), int(result['pair'][1])] = result['acc_train']
            test_img[int(result['pair'][0]), int(result['pair'][1])] = result['acc_test']
            train_img[int(result['pair'][1]), int(result['pair'][0])] = result['acc_train']
            test_img[int(result['pair'][1]), int(result['pair'][0])] = result['acc_test']
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        ax[0].imshow(train_img, cmap='hot', interpolation='nearest')
        ax[1].imshow(test_img, cmap='hot', interpolation='nearest')
        ax[0].set_title("TRAIN")
        ax[1].set_title("TEST")
        fig.suptitle("K-Means Results %i samples/digit\n(mean accuracy over %i trials)" %
                     (self._data.get_n()[0], self._n_KM_trials))
        # colorbar
        fig.colorbar(ax[0].imshow(train_img, cmap='hot', interpolation='nearest'), ax=ax[0])
        fig.colorbar(ax[1].imshow(test_img, cmap='hot', interpolation='nearest'), ax=ax[1])

    def plot_pairwise_clusterings(self, ax, results, which='train'):
        """
        Plot the clustering results for each digit pair.

        """
        v_digits = []
        h_digits = []

        for result in results:
            pair = result['pair']

            data = self._get_train_test_data(pair)
            if which == 'train':
                pred_labels = result['train_out']
                true_labels = data['y_train']
                points = data['x_train']
            else:
                pred_labels = result['test_out']
                true_labels = data['y_test']
                points = data['x_test']
            unit_points = project_binary_clustering(points, true_labels)
            points_shifted = unit_points + np.array(pair)
            # if pair==(2,3):
            #    import ipdb; ipdb.set_trace()
            plot_binary_clustering(ax, points_shifted, pred_labels.astype(int), true_labels.astype(int),
                                   point_size=2, circle_size=15)
            v_digits.append(pair[1])
            h_digits.append(pair[0])
        v_digits = np.sort(np.unique(v_digits))
        h_digits = np.sort(np.unique(h_digits))

        # draw black lines between pairs
        x_lim, y_lim = [0, 9], [1, 10]
        for i in range(9):
            if i < 9:
                ax.plot([i, i], y_lim, color='black', linewidth=0.5)
            ax.plot(x_lim, [i+1, i+1], color='black', linewidth=0.5)
        # set x-ticks and y-ticks for all integers
        ax.set_xticks(np.array(h_digits)+.5)
        ax.set_yticks(np.array(v_digits)+.5)
        ax.set_xticklabels(["%i" % i for i in h_digits])
        ax.set_yticklabels(["%i" % i for i in v_digits])

        ax.xaxis.tick_top()
        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)

    def _plot_best_and_worst_pairs(self, results, n=2, title=""):
        # only training for now

        def _show_pair(fig, ax, result, title_2, index):
            pair = result['pair']
            data = self._get_train_test_data(pair)
            show_digit_cluster_collage(ax,
                                       data['img_train'],
                                       data['x_train'],
                                       result['train_out'],
                                       data['y_train'],
                                       max_n_imgs=300)
            fig.suptitle("%s - %s pairs, showing %i of %i:  %s" % (title, title_2, index+1, n, pair))

        best = sorted(results, key=lambda x: x['acc_train'], reverse=True)[:n]
        worst = sorted(results, key=lambda x: x['acc_train'])[:n]

        for i, res in enumerate(best):
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            _show_pair(fig, ax, res, "Best", i)
            plt.show()

        for i, res in enumerate(worst):
            fig, ax = plt.subplots(1, 2, figsize=(12, 6))
            _show_pair(fig, ax, res, "Worst", i)
            plt.show()

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

    def _get_fisher_results(self):
        """
        Apply fisher's linear discriminant analysis to the data as an
        upper bound to the performance of K-means, since K-means with 2
        classes has a linear decision boundary, and LDA finds the optimal one.
        """
        if not os.path.exists(self.FISHER_CACHE_FILE):
            work = []
            for pair in self._digit_pairs:
                data = self._get_train_test_data(pair)
                work.append({'pair': pair,
                            'graph_name': 'Fisher LDA',
                             'data': data})
            output = [_test_fisher(w) for w in work]
            # if self._n_cpu == 1:
            #    output = [_test_fisher(w) for w in work]
            # else:
            #    with Pool(self._n_cpu) as pool:
            #        output = pool.map(_test_fisher, work)
            logging.info("Saving Fisher results to %s" % self.FISHER_CACHE_FILE)
            with open(self.FISHER_CACHE_FILE, 'wb') as f:
                pickle.dump(output, f)

        else:
            logging.info("Reading Fisher results from %s" % self.FISHER_CACHE_FILE)
            with open(self.FISHER_CACHE_FILE, 'rb') as f:
                output = pickle.load(f)

        return output

    def run(self):
        self._fisher_results = self._get_fisher_results()
        self._k_means_results = self._get_kmeans_results()
        self._k_means_stats = self._stats_from_results(self._k_means_results)
        #_, ax = plt.subplots(1, 1)
        #_, ax2 = plt.subplots(1, 1)

        #self.plot_pairwise_clusterings(ax, self._k_means_results, 'train')
        # self.plot_pairwise_clusterings(ax2, self._fisher_results, 'train')

        # self._spectral_results = self._get_spectral_results()
        # self.plot_km_digit_confusion()

        self._plot_best_and_worst_pairs(self._fisher_results, n=10)
        self._plot_best_and_worst_pairs(self._k_means_results, n=10)

        logging.info("KMeans results:")
        pprint(self._k_means_stats)
        # fig, ax = plt.subplots()
        # self._plot_spectral_result(ax, 'n-neighbors', 'train')
        #ax.set_title("K-Means cluster separation")
        #ax2.set_title("Fisher LDA class separation")
        plt.show()

    def _plot_kmean_baseline(self, ax, test_train='test'):
        x = ax.get_xlim()
        means = self._k_means_stats['test_mean'] if test_train == 'test' else self._k_means_stats['train_mean']
        sd = self._k_means_stats['test_sd'] if test_train == 'test' else self._k_means_stats['train_sd']
        self._plot_bands(ax, x, [means, means], [sd, sd], "K-Means Baseline")
        ax.set_xlim(x[0], x[1])

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


def _test_kmeans(work):
    """
    multiprocessing helper for testing K-means, n-trials on a single digit pair.
    """
    pair = work['pair']
    data = work['data']
    n_trials = work['n_trials']

    best_results = {'acc_train': 0,
                    'acc_test': 0,
                    'train_out': None,
                    'test_out': None}
    for _ in range(n_trials):
        km = KMeansAlgorithm(2)
        km.fit(data['x_train'])
        classifier = ClusterClassifier(km, data['x_train'], data['y_train'])
        train_out = classifier.predict(data['x_train'])
        test_out = classifier.predict(data['x_test'])
        train_acc = np.mean(train_out == data['y_train'])
        test_acc = np.mean(test_out == data['y_test'])

        if train_acc > best_results['acc_train']:
            best_results = {'acc_train': train_acc,
                            'acc_test': test_acc,
                            'train_out': train_out,
                            'test_out': test_out}

    best_results['pair'] = pair
    best_results['graph_name'] = 'k-means'
    print("Tested K-Means for digit pair {}, {} samples, {} times, best results: train_acc={}, test_acc={}".format(
        pair, len(data['x_train']), n_trials, train_acc, test_acc))

    return best_results


def _test_fisher(work):
    """
    multiprocessing helper for testing Fisher's LDA on a single digit pair.
    :param work: dict with {
                    'pair': (a, b),
                    'graph_name': 'Fisher LDA',
                    'data': dict with x_train, x_test, y_train, y_test}
    :returns: list of dicts with {
                    'acc_train': train accuracy,
                    'acc_test': test accuracy,
                    'pair': (a, b),
                    'graph_name': 'Fisher LDA'}
    """

    pair = work['pair']
    data = work['data']
    model = FisherLDA()
    model.fit(data['x_train'], data['y_train'])
    classif = ClusterClassifier(model, data['x_train'], data['y_train'])
    train_out = classif.predict(data['x_train'])
    test_out = classif.predict(data['x_test'])

    train_out = classif.predict(data['x_train'])
    test_out = classif.predict(data['x_test'])
    train_acc = np.mean(train_out == data['y_train'])
    test_acc = np.mean(test_out == data['y_test'])

    print("Tested Fisher LDA on digits %s:  (Train acc:  %.4f, Test acc: %.4f)" % (
        pair, train_acc, test_acc))

    return {'acc_train': train_acc,
            'acc_test': test_acc,
            'test_out': test_out,
            'train_out': train_out,
            'pair': pair,
            'graph_name': 'Fisher LDA'}


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
