"""
Common functions for clustering MNIST data to classify digits.

The general process is:
  1. Reduce dimensionality from 28x28 images to 30 using PCA.
  2. Cluster the data using spectral clustering / k-means.
  3. Assign cluster ids to class labels to maximize accuracy.
"""
import logging
import argparse
import numpy as np
from similarity import FullSimGraph, NNSimGraph, SoftNNSimGraph, EpsilonSimGraph
from munkres import Munkres


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


GRAPH_TYPES = {'full': FullSimGraph,
               'n-neighbors': NNSimGraphNonMutual,
               'n-neighbors_mutual': NNSimGraphMutual,
               'soft_neighbors_additive': SoftNNSimGraphAdditive,
               'soft_neighbors_multiplicative': SoftNNSimGraphMultiplicative,
               'epsilon': EpsilonSimGraph}


GRAPH_PARAM_NAMES = {'full': 'sigma',
                     'n-neighbors': 'k',
                     'n-mutual-neighbors': 'k',
                     'soft_neighbors_additive': 'alpha',
                     'soft_neighbors_multiplicative': 'alpha',
                     'epsilon': 'epsilon'}
    

class MNISTResult(object):
    """
    Class to hold results of a single clustering.
    If true labels are provided, will compute cluster ID to class label mapping
     and will computer accuracy.  (Keep classes balanced for best results.)
    """
    def __init__(self, model, data, true_labels=None, sample_indices=None, aux=None):
        """
        :param model: ClusteringAlgorithm, with fit() called.
        :param data: data used to fit the model
        :param cluster_ids: cluster ids for each data point
        :param true_labels: true labels for each data point
        :param sample_indices: indices into full NIST dataset used to train model
        :param aux: any auxiliary data to store with the result
        """
        self.aux = aux
        self.k=model.get_k()
        #self.data = data  # don't bother saving, just save indices
        self.inds = sample_indices
        self.cluster_ids = model.assign(data)
        self.true_labels = true_labels
        if true_labels is not None:
            self.pred_labels = self._get_cluster_labels()
            self.accuracy = self._get_accuracy()

    def _get_cluster_labels(self):
        if self.k == 2:
            # just see which is better
            err_rate = np.mean(self.cluster_ids != self.true_labels)
            if err_rate > 0.5:
                return 1 - self.cluster_ids
            else:
                return self.cluster_ids
        else:
            # use the Hungarian algorithm to assign cluster labels to digit labels
            m = Munkres()
            cost_matrix = np.zeros((self.k, self.k))
            for i in range(self.k):
                for j in range(self.k):
                    cost_matrix[i, j] = -np.sum((self.pred_labels == i) & (self.true_labels == j))
            indexes = m.compute(cost_matrix)
            cluster_label_map = {i: j for i, j in indexes}
            return cluster_label_map

    def _get_accuracy(self):
        if self.true_labels is None:
            raise ValueError("True labels are not provided.")
        return np.mean(self.pred_labels == self.true_labels)
