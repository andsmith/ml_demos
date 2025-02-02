"""
Get baseline results using K-means on the MNIST dataset.
(also Fisher LDA)
"""

import logging
import numpy as np
from sklearn.cluster import KMeans
from mnist_common import MNISTResult
from mnist_data import MNISTData
from clustering import KMeansAlgorithm
from mnist_plots import (plot_pairwise_digit_confusion, plot_pairwise_clusterings,
                         plot_extreme_pairs, plot_full_confusion, plot_full_embedding)
import pickle
from multiprocessing import Pool, cpu_count
import os
from util import load_cached
import sys
from fisher_lda import FisherLDA

import matplotlib.pyplot as plt

# Common params for full and pairwise experiments
DIM = 30
N_REP = 10
N_SAMP = 1000
N_TEST = 800
N_BOOT = 50
N_CPU = 1


class KMeansFull(object):
    def __init__(self, dim=DIM, n_rep=N_REP, n_samp=(N_SAMP, N_TEST), n_bootstraps=N_BOOT, n_cpu=N_CPU):
        """
        :param dim: PCA dimension
        :param n_rep: Run KMeans this many times to get best result over random inits.
        :param n_samp: number of samples per digit for training
        :param n_bootstraps: number of bootstraps for error estimation
        :param n_cpu: number of CPUs to use for parallel processing
        """
        self._n_rep = n_rep
        self._dim = dim
        self._n_samples, self._n_test = n_samp
        self._n_bootstraps = n_bootstraps
        self._n_cpu = cpu_count()-4 if n_cpu is None else n_cpu
        self._data = MNISTData(pca_dim=dim)
        self._all_results = self._init_data()

    def _init_data(self):
        return load_cached(self._get_results, "KMeans_full_r=%i_n=%i_b=%i.pkl" % (
            self._n_rep, self._n_samples, self._n_bootstraps))  # add n-test to filename?

    def _get_work(self, name):
        """
        Work is divided over bootstraps, each process gets a random sample to run n_rep times.
        :returns: list of work, dict with {'graph_name': 'kmeans',
                                           'data': (x, y),  # NxD, N 
                                           'inds': random sample indices,
                                           'n_trials': n_rep,
                                           'n_samp': n_samp}
        """
        work = []
        for i in range(self._n_bootstraps):
            sample_data = self._data.get_sample(self._n_samples, self._n_test)
            work.append({'graph_name': name,
                         'data': sample_data,
                         'aux': {'trial': i, 'n_trials': self._n_bootstraps},
                         'n_trials': self._n_rep})
        return work

    def _get_results(self):
        work = self._get_work("K-Means")
        if self._n_cpu == 1:
            logging.info("Running K-Means results in single process: %i tasks." % len(work))
            results = [_test_kmeans(w) for w in work]
        else:
            logging.info("Running K-Means results in %i processes:  %i tasks." % (self._n_cpu, len(work)))
            with Pool(self._n_cpu) as pool:
                results = pool.map(_test_kmeans, work)
        return results

    def plot_results(self):
        """
        Plot 1:  mean confusion matrix over all bootstraps on the left, histogram of accuracies on the right.
        Plot 2:  Color-coded embedding of all digits, using best result
        """
        # show embedding
        # title = "KMeans 10-digit embedding, PCA dim=%i\n(%i samples/digit, %i trials)\ncolor by cluster ID" % (
        #    self._dim, self._n_samples, self._n_bootstraps)
        # plot_full_embedding(self._all_results, self._data, title, max_n_imgs=200, image_extent_frac=0.015)
        # calls plt.show() at the end

        # show confusion matrices & histograms in one figure
        fig, ax = plt.subplots(2, 2, figsize=(5, 8))

        #    train
        def _show_confusion(ax, results, which):
            img = plot_full_confusion(ax, results, which=which)
            ax.set_ylabel("Cluster ID")
            ax.set_xlabel("True digit")
            ax.set_title("confusion - %s data" % which)
            fig.colorbar(img, ax=ax)

        _show_confusion(ax[0][0], self._all_results, 'train')
        _show_confusion(ax[0][1], self._all_results, "test")

        # accuracy histograms:
        def _show_hist(ax, accuracies, title):
            counts, bins = np.histogram(accuracies, bins=15)
            bin_centers = (bins[:-1] + bins[1:]) / 2.
            ax.plot(bin_centers, counts, 'o-')
            ax.set_ylabel("count")
            ax.set_xlabel("Histogram of accuracy")

        accuracies_test = [result.accuracy['test'] for result in self._all_results]
        accuracies_train = [result.accuracy['train'] for result in self._all_results]
        _show_hist(ax[1][0], accuracies_train, "Training data")
        _show_hist(ax[1][1], accuracies_test, "Test data")

        fig.suptitle("KMeans 10-digit classification, PCA dim=%i\n(%i samples/digit, %i trials)" % (
            self._dim, self._n_samples, self._n_bootstraps))


class KMeansPairwise(object):
    def __init__(self, dim=DIM, n_rep=N_REP, n_samp=N_SAMP, n_cpu=N_CPU):
        self._n_rep = n_rep
        self._dim = dim
        self._n_samples = n_samp
        self._n_cpu = cpu_count()-4 if n_cpu is None else n_cpu
        self._data = MNISTDataPCA(dim=dim)
        self._digit_pairs = [(i, j) for i in range(10) for j in range(i+1, 10)]
        self._all_results = self._init_data()

    def _init_data(self):
        return load_cached(self._get_results, "KMeans_pairwise_r=%i_n=%i.pkl" % (
            self._n_rep, self._n_samples))

    def _get_work(self, name):
        work = []
        for pair in self._digit_pairs:
            # Tasks will be divided up by each digit pair.
            # sample randomly from each digit, each time.
            inds0 = self._data.get_digit_sample_inds(pair[0], self._n_samples)
            inds1 = self._data.get_digit_sample_inds(pair[1], self._n_samples)

            x = np.vstack((self._data.get_digit(pair[0])[inds0],
                           self._data.get_digit(pair[1])[inds1]))
            y = np.hstack((np.zeros(inds0.size),
                           np.ones(inds1.size)))

            work.append({'graph_name': name,
                         'aux': {'pair': pair},
                         'inds': (inds0, inds1),
                         'data': (x, y),
                         'n_trials': self._n_rep})
        return work

    def _get_results(self):
        work = self._get_work("K-Means")
        if self._n_cpu == 1:
            logging.info("Running K-Means results in single process.")
            results = [_test_kmeans(w) for w in work]
        else:
            logging.info("Running K-Means results in %i processes." % self._n_cpu)
            with Pool(self._n_cpu) as pool:
                results = pool.map(_test_kmeans, work)
        return results

    def plot_results(self, prefix="KMeans"):

        # show best & worst pairs
        title = "%s(pca=%i)" % (prefix, self._dim)
        fig, ax = plot_extreme_pairs(self._all_results, self._data, n=3, title=title)

        # show pairwise clusterings
        fig, ax = plt.subplots(figsize=(7, 6))
        plot_pairwise_clusterings(ax, self._all_results, self._data)
        ax.set_title("%s Pairwise Clustering (PCA dim=%i)\n(%i samples/digit, best of %i trials)" % (
            prefix, self._dim, self._n_samples, self._n_rep))
        plt.tight_layout()
        # plt.show()

        # show confusion matrix
        fig, ax = plt.subplots()
        img = plot_pairwise_digit_confusion(ax, self._all_results)
        ax.set_title("%s pairwise accuracy (PCA dim=%i)\n(%i samples/digit, best of %i trials)" % (
            prefix, self._dim, self._n_samples, self._n_rep))
        fig.colorbar(img, ax=ax)
        # plt.show()


def _test_kmeans(work):
    """
    multiprocessing helper for testing K-means, n-trials on a single digit pair.
    :param work: dict with {'graph_name': 'kmeans',
                            'data': MNistSample
                            'n_trials': n_rep}
    """
    # import ipdb; ipdb.set_trace()
    n_trials = work['n_trials']

    x, y = work['data'].get_data('train')
    k = np.unique(y).size
    best_result = None
    for _ in range(n_trials):
        km = KMeansAlgorithm(k)
        km.fit(x)
        result = MNISTResult(k, km, work['data'])
        if best_result is None or result.accuracy['train'] > best_result.accuracy['train']:
            best_result = result

    if work['aux'] is not None:
        print("Tested K-Means({}):  {} samples, {} times, best results: info={}.  accuracy={}".format(
            k, len(y), n_trials, work['aux'], best_result.accuracy))

    return best_result


class FisherPairwise(KMeansPairwise):
    def __init__(self, dim=DIM, n_samp=N_SAMP):
        super().__init__(dim, 0, n_samp, n_cpu=1)

    def _init_data(self):
        return load_cached(self._get_fisher_results, "Fisher_pairwise_n=%i.pkl" % (self._n_samples))

    def _get_fisher_results(self):
        work = self._get_work("Fisher LDA")

        # no need for multiprocessing
        results = [_test_fisher(w) for w in work]
        return results

    def plot_results(self, prefix="Fisher LDA"):
        super().plot_results(prefix=prefix)


def _test_fisher(work):
    data = work['data']
    model = FisherLDA()
    model.fit(data[0], data[1])
    result = MNISTResult(2, model, data[0], data[1], sample_indices=work['inds'], aux=work['aux'])

    print("Tested Fisher LDA on digits %s:  (Accuracy:  %s)" % (
        work['aux'], result.accuracy))
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Running KMeansPairwise.")

    # import ipdb; ipdb.set_trace()

    # km = KMeansPairwise()
    # km.plot_results()
    # plt.show()

    km = KMeansFull()
    km.plot_results()
    # plt.show()

    # f = FisherPairwise()
    # f.plot_results()
    plt.show()
