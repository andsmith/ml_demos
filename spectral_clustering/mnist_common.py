"""
Common functions for clustering MNIST data to classify digits.

The general process is:
  1. Reduce dimensionality from 28x28 images to 30 using PCA.
  2. Cluster the data using spectral clustering / k-means.
  3. Assign cluster ids to class labels to maximize accuracy.
"""
import logging
import numpy as np
from similarity import FullSimGraph, NNSimGraph, SoftNNSimGraph, EpsilonSimGraph
from munkres import Munkres
import json


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


GRAPH_TYPES = {'n-neighbors_mutual': NNSimGraphMutual,
               'n-neighbors': NNSimGraphNonMutual,
               'soft_neighbors_additive': SoftNNSimGraphAdditive,
               'soft_neighbors_multiplicative': SoftNNSimGraphMultiplicative,
               'full': FullSimGraph,
               'epsilon': EpsilonSimGraph}


GRAPH_PARAM_NAMES = {'full': 'sigma',
                     'n-neighbors': 'k',
                     'n-neighbors_mutual': 'k',
                     'soft_neighbors_additive': 'alpha',
                     'soft_neighbors_multiplicative': 'alpha',
                     'epsilon': 'epsilon'}


class MNISTResult(object):
    """
    Class to hold results of a single clustering.
    If true labels are provided, will compute cluster ID to class label mapping
     and will computer accuracy.  (Keep classes balanced for best results.)
    """

    def __init__(self, k, model, data, cluster_ids=None):
        """
        :param model: ClusteringAlgorithm, with fit() called.
        :param data: MNISTSamples object
        :param sample_indices: indices into full NIST dataset used to train model
            (dict: key=digit, value = index array)
        """
        # save these for results reporting
        self.k = k
        self.digits = np.array(data.digits)
        self.pca_dim = data.pca_dim
        self.inds = {'train': data.train_inds,
                     'test': data.test_inds}
        self.pca_transf = data.pca_transf  # for transforming new data

        x_test, y_test = data.get_data('test')
        x_train, y_train = data.get_data('train')

        # get cluster Id's from training set, derive mapping to digit labels, then do test set
        self.cluster_ids = {'train': model.assign(x_train),
                            'test': model.assign(x_test)} if cluster_ids is None else cluster_ids
        self.label_map = self._get_cluster_labels(self.cluster_ids['train'], y_train)
        self.pred_labels = {'train': self.label_map[self.cluster_ids['train']],
                            'test': self.label_map[self.cluster_ids['test']]}
        self.true_labels = {'train': y_train,
                            'test': y_test}

        self.accuracy = self._get_accuracy()

        self._info = {}  # for storing additional plotting info

    def get_info(self, name):
        if name not in self._info:
            return None
        return self._info.get(name, None)

    def set_info(self, name, value):
        self._info[name] = value

    def _get_cluster_labels(self, ids, labels):
        """
        Cluster IDs are in [0, k-1], labels are one of self.digits,
        so we need to assign each cluster to a label.
        We find the bijective mapping that minimizes the error rate.
        :returns: array of length k, where element i is the digit label for cluster i.
        """
        if self.k == 2:
            label_inds = (labels == self.digits[1]).astype(int)
            err_rate = np.mean(ids != label_inds)
            if err_rate > 0.5:
                return self.digits[::-1]
            else:
                return self.digits
        else:
            # use the Hungarian algorithm to assign cluster labels to digit labels
            m = Munkres()
            cost_matrix = np.zeros((self.k, self.k))
            for i in range(self.k):
                for j in range(self.k):
                    cost_matrix[i, j] = -np.sum((ids == i) & (labels == j))
            indexes = m.compute(cost_matrix)
            cluster_label_map = np.zeros(self.k, dtype=int)
            for row, column in indexes:
                cluster_label_map[row] = column

            return cluster_label_map

    def _get_accuracy(self):
        return {'train': np.mean(self.pred_labels['train'] == self.true_labels['train']),
                'test': np.mean(self.pred_labels['test'] == self.true_labels['test'])}

    def _get_confusion_matrix(self, which='test'):
        """
        element i,j is the fraction of samples with label j were correctly assigned to cluster i
        """
        def _get_mat(true_labels, pred_labels):
            conf_mat = np.zeros((self.k, self.k))
            for i in range(self.k):
                for j in range(self.k):
                    conf_mat[i, j] = np.sum((pred_labels == j) & (true_labels == i)) / \
                        np.sum(true_labels == j)
            return conf_mat
        if which == 'train':
            return _get_mat(self.true_labels['train'], self.pred_labels['train'])
        else:
            return _get_mat(self.true_labels['test'], self.pred_labels['test'])


def make_fake_pairwise_result(sample, pair, n_bad=20):
    """
    Create a fake pairwise result for testing.
    :param sample: MNISTSample object
    :param pair: tuple of digits to compare
    :param n_bad: number of bad pairs to add (randomly)
    :returns: MNISTResult with all correct pred_labels in test and train, but n_bad of each class.
    """
    def _make_pred_labels(which):
        src = sample.test if which == 'test' else sample.train
        labels0 = np.zeros(src[pair[0]].shape[0], dtype=int)
        labels0[np.random.choice(labels0.size, n_bad, replace=False)] = 1
        labels1 = np.ones(src[pair[1]].shape[0], dtype=int)
        labels1[np.random.choice(labels1.size, n_bad, replace=False)] = 0
        return np.hstack([labels0, labels1])

    cluster_ids = {'train': _make_pred_labels('train'),
                   'test': _make_pred_labels('test')}
    result = MNISTResult(2, None, sample, cluster_ids)
    return result


def test_fake_result():
    import mnist_data
    data = mnist_data.MNISTData(pca_dim=30)
    sample = data.get_sample(100, 50, [5, 8])

    res = make_fake_pairwise_result(sample, (5, 8), 5)
    print(res.accuracy)
    assert res.accuracy['train'] == 0.95, "Train accuracy should be 0.95, but was %f" % res.accuracy['train']
    assert res.accuracy['test'] == 0.9, "Test accuracy should be 0.9, but was %f" % res.accuracy['test']


baseline_filename = "KM_baselines.json"


class Baselines(object):
    """
    For adding K-means/fisher accuracies to other plots for comparison.
    """

    def __init__(self, file=baseline_filename):
        self._file = file
        raw = self._load()
        self.data = {'fisher': self._decode_pairwise(raw['fisher_pairwise']),
                    'pairwise': self._decode_pairwise(raw['km_pairwise']),
                    'full': raw['km_full']}
    
    def _decode_pairwise(self, raw):
        # split key from float A.B  to (A, B)
        def str_to_pair(s_pair):
            p = float(s_pair)
            return int(p), int((p - int(p)) * 10)
        
        return {str_to_pair(k): v for k, v in raw.items()}
    

    def _load(self):
        with open(self._file, 'r') as f:
            items= json.load(f)
        logging.info("Loaded baselines for: [%s]" % ', '.join(items.keys()))
        return items



def test_baselines():
    b= Baselines()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    test_baselines()
    print("Done.")
