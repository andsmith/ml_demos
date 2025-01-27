"""
Get baseline results using K-means on the MNIST dataset.
(also Fisher LDA)
"""

import logging
import numpy as np
from sklearn.cluster import KMeans
from mnist_common import MNISTResult
from mnist_data import MNISTDataPCA
from clustering import KMeansAlgorithm
from mnist_plots import plot_pairwise_digit_confusion, plot_pairwise_clusterings, plot_best_and_worst_pairs
import pickle
from multiprocessing import Pool, cpu_count
import os
from util import load_cached
import sys
from fisher_lda import FisherLDA

import matplotlib.pyplot as plt


class KMeansPairwise(object):
    def __init__(self, dim=30, n_rep=5, n_samp=1000, n_cpu=1):
        self._n_rep = n_rep
        self._dim = dim
        self._n_samples = n_samp
        self._n_cpu = cpu_count()-4 if n_cpu is None else n_cpu
        self._data = MNISTDataPCA(dim=dim)
        self._digit_pairs = [(i, j) for i in range(10) for j in range(i+1, 10)]
        self._all_results = self._init_data()

    def _init_data(self):
        return load_cached(self._get_kmeans_results, "KMeans_pairwise_r=%i_n=%i.pkl" % (
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
                         'pair': pair,
                         'inds': (inds0, inds1),
                         'data': (x, y),
                         'n_trials': self._n_rep})
        return work

    def _get_kmeans_results(self):
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
        # show confusion matrix
        fig, ax = plt.subplots()
        img = plot_pairwise_digit_confusion(ax, self._all_results)
        ax.set_title("%s Pairwise Accuracy (PCA dim=%i)\n(%i samples/digit, %i trials)" % (
            prefix,self._dim, self._n_samples, self._n_rep))
        fig.colorbar(img, ax=ax)
        plt.show()

        # show pairwise clusterings
        fig, ax = plt.subplots()
        plot_pairwise_clusterings(ax, self._all_results, self._data)
        ax.set_title("%s Pairwise Clustering (PCA dim=%i)\n(%i samples/digit, %i trials)" % (
            prefix,self._dim, self._n_samples, self._n_rep))
        plt.show()

        # show best & worst pairs
        plot_best_and_worst_pairs(self._all_results, self._data, 2, prefix)


def _test_kmeans(work):
    """
    multiprocessing helper for testing K-means, n-trials on a single digit pair.
    """
    data = work['data']
    n_trials = work['n_trials']

    best_result = None
    for _ in range(n_trials):
        km = KMeansAlgorithm(2)
        km.fit(data[0])
        x, y = data
        result = MNISTResult(km, x, y,
                             sample_indices=work['inds'],
                             aux={'pair': work['pair']})
        if best_result is None or result.accuracy > best_result.accuracy:
            best_result = result

    print("Tested K-Means:  {} samples, {} times, best results {}: accuracy={}, test_acc={}".format(
        len(data[1]), n_trials, work['pair'], best_result.accuracy, best_result.accuracy))

    return best_result




class FisherPairwise(KMeansPairwise):
    def __init__(self, dim=30, n_samp=3000):
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
    pair = work['pair']
    data = work['data']
    model = FisherLDA()
    model.fit(data[0], data[1])
    result = MNISTResult(model, data[0], data[1], sample_indices=work['inds'], aux={'pair': pair})

    print("Tested Fisher LDA on digits %s:  (Accuracy:  %.4f)" % (
        pair, result.accuracy))
    return result


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Running KMeansPairwise.")
    #km = KMeansPairwise()
    #km.plot_results()

    f = FisherPairwise()
    f.plot_results()
