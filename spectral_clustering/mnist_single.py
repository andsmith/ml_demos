"""
What sub-types of individual digits are there in the MNIST dataset?
Can the digits with sub-types be automatically distinguished from digits all drawn the same way?

To investigate, cluster each digit with varying K and, for each digit, plot::
  * The clusters in a 2-d embedding that keeps cluster centers far apart. (2 principle components)
  * Plot lowest 2K eigenvalues next to the embedding.
  * Show each cluster's "prototype", the sample closest to its center of mass in eigenspace.
"""

from mnist_data import MNISTDataPCA
from clustering import SpectralAlgorithm
from mnist_common import MNISTResult, GRAPH_TYPES, GRAPH_PARAM_NAMES

class DigitClustering(object):
    """
    Each experiment will compute the similarity graph, 
    plot the spectrum, and let the user choose a K to cluster it.
    Results are displayed as plots of the clusters in 2d embeddings.
    """
    def __init__(self, graph_name, graph_args, n_samp=5000, dim=30):
        """
        :param graph_name: string, type of similarity graph to use
        :param graph_args: dict, arguments to pass to similarity graph constructor
        :param n_samp: number of samples per digit for training
        :param dim: PCA dimension
        """
        self._n_samples = n_samp
        self._dim = dim
        self._data = MNISTDataPCA(dim=dim)
        self._graph_name = graph_name
        self._graph_args = graph_args