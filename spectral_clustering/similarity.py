from clustering import ClusteringAlgorithm
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
import cv2
from enum import IntEnum
from util import image_from_floats, apply_colormap
from abc import ABC, abstractmethod
from scipy.spatial import KDTree


class SimilarityGraphTypes(IntEnum):
    EPSILON = 0
    NN = 1
    FULL = 2
    SOFT_NN =3


def get_kind_from_name(names, name):
    for kind, kind_name in names.items():
        if kind_name == name:
            return kind
    raise ValueError(f"Invalid name: {name}")


class SimilarityGraph(object):
    """
    Build a similarity graph from a set of points using euclidean distances.
    """

    def __init__(self, points):
        """
        Construct a similarity graph from points in the unit square.
        :param points: 2D numpy array of points
        :param colormap: colormap to use for the similarity matrix image
        """
        self._points = points
        self._mat = self._build()
        # print("Built similarit matrix, weights in range [%f, %f], frac_nonzero=%.5f" % (
        # np.min(self._mat), np.max(self._mat), np.count_nonzero(self._mat) / self._mat.size))

    @abstractmethod
    def _build(self):
        """
        Build the similarity matrix.
        :return: similarity matrix, image of the similarity matrix
        """
        pass

    def draw_graph(self, img):
        """
        Draw an edge between points with nonzero entries in the similarity matrix.
        """
        lines = []
        nonzero = np.nonzero(np.triu(self._mat))
        for i, j in zip(*nonzero):
            lines.append(np.array([self._points[i], self._points[j]], dtype=np.int32))

        cv2.polylines(img, lines, False, (128, 128, 128), 1, cv2.LINE_AA)

    def get_matrix(self):
        return self._mat

    @abstractmethod
    def make_img(self,colormap=None):
        pass

    def get_tree(self):
        return KDTree(self._points)

class EpsilonSimGraph(SimilarityGraph):

    def __init__(self, points, epsilon):
        self._epsilon = epsilon
        super().__init__(points)

    def _build(self):
        """
        Build the similarity matrix using epsilon distance.
        I.e., two points are connected if their distance is less than epsilon, all 
        weight 1 or 0.
        """
        dists = squareform(pdist(self._points))
        np.fill_diagonal(dists, np.inf)
        sim_matrix = np.zeros(dists.shape)
        sim_matrix[dists <= self._epsilon] = 1
        return sim_matrix
    
    def make_img(self,colormap=None):
        
        img = image_from_floats(self._mat, 0, 1)
        img = cv2.merge([img, img, img])
        return img


class FullSimGraph(SimilarityGraph):
    def __init__(self, points, sigma):
        self._sigma = sigma
        super().__init__(points)

    def _build(self):
        """
        Build the similarity matrix using the full similarity function.
        """
        dists = squareform(pdist(self._points))
        np.fill_diagonal(dists, np.inf)
        sim_matrix = np.exp(-dists**2 / (2*self._sigma**2))
        return sim_matrix
    
    def make_img(self,colormap=None):
        img = apply_colormap(self._mat, colormap)
        return img
        
class SoftNNSimGraph(SimilarityGraph):
    """
    Construct a similarity graph S[i,j]:
       
       Let R[i,j] be the rank of point j in the sorted list of distances from point i, and
       let W[i,j] = R[i,j]) ^ -alpha be the weight of the directed edge from i to j.
       
       Then the similarity between i and j is given by:

           S[i,j] = W[i,j] + W[j,i] if additive, or 
                  = W[i,j] * W[j,i] if multiplicative.
          
    """
    
    def __init__(self, points, alpha, additive=True):
        """
        Construct a similarity graph using K-nearest neighbors.
        :param points: 2D numpy array of points
        :param k: number of nearest neighbors
        :param alpha: exponent for the weight function
        :param additive: if True, use additive weights, otherwise use multiplicative weights
        """
        self._alpha = alpha
        self._additive = additive
        super().__init__(points)

    def _build(self):
        dists = squareform(pdist(self._points))
        np.fill_diagonal(dists, np.inf)
        orders = np.argsort(dists, axis=1)
        weights = np.zeros_like(dists)
        n = dists.shape[0]
        for row in range(n):
            weights[row,orders[row,:-1]] = np.exp(-np.arange(1, n) **2.0 /  self._alpha**2.0)

        
        if self._additive:
            sim_matrix = weights + weights.T
        else:
            sim_matrix = weights * weights.T

        return sim_matrix
    
    def make_img(self,colormap=None):
        img = apply_colormap(self._mat, colormap)
        return img

class NNSimGraph(SimilarityGraph):
    def __init__(self, points, k, mutual):
        """
        Construct a similarity graph using K-nearest neighbors.
        :param points: 2D numpy array of points
        :param k: number of nearest neighbors
        :param mutual: if True, only connect if both points are among each other's K-nearest neighbors,
            otherwise connect if either is among the other's K-nearest neighbors.
        """
        self._k = k
        self._mutual = mutual
        super().__init__(points)

    def _build(self):
        """
        Build the similarity matrix using K-nearest neighbors, i.e.
        two points are connected if they are among each other's K-nearest neighbors.
        :param k: number of nearest neighbors
        :param mutual: if True, only connect if both points are among each other's K-nearest neighbors,
            otherwise connect if either is among the other's K-nearest neighbors.
        """
        # add 1 to k to include self
        nbrs = NearestNeighbors(n_neighbors=self._k+1, algorithm='ball_tree').fit(self._points)
        edge_mat = nbrs.kneighbors_graph(self._points, mode='connectivity').toarray()

        if self._mutual:
            edge_mat = np.logical_and(edge_mat, edge_mat.T)
        else:
            edge_mat = np.logical_or(edge_mat, edge_mat.T)
        np.fill_diagonal(edge_mat, 0) 
        return edge_mat
    
    def make_img(self, colormap=None):
        img = image_from_floats(self._mat, 0, 1)
        img = cv2.merge([img, img, img])
        return img


# labels for slider param for different simgraph types
SIMGRAPH_PARAM_NAMES = {SimilarityGraphTypes.NN: "N-nearest",
                        SimilarityGraphTypes.SOFT_NN: "Alpha",
                        SimilarityGraphTypes.EPSILON: "Epsilon",
                        SimilarityGraphTypes.FULL: "Sigma"}
# simgraph type menu options
SIMGRAPH_KIND_NAMES = {SimilarityGraphTypes.NN: "N-nearest",
                       SimilarityGraphTypes.SOFT_NN: "A-nearest",
                       SimilarityGraphTypes.EPSILON: "Epsilon",
                       SimilarityGraphTypes.FULL: "Full"}


def test_soft_nn_sim():
    points = np.random.rand(1000, 2)
    sim_graph_add = SoftNNSimGraph(points, alpha=.01, additive=True)
    sim_graph_mul = SoftNNSimGraph(points, alpha=.01, additive=False)
   
    import matplotlib.pyplot as plt
    plt.subplot(2,2,1)
    plt.imshow(sim_graph_add.make_img())
    plt.subplot(2,2,2)
    plt.imshow(sim_graph_mul.make_img())
    plt.subplot(2,2,3)
    for i in range(points.shape[0]):
        # plot each with its index
        plt.text(points[i,0], points[i,1], str(i), fontsize=12, ha='center')
    plt.subplot(2,2,4)
    counts, bins = np.histogram(sim_graph_add.get_matrix().flatten(), bins=50)
    plt.plot((bins[:-1] + bins[1:])/2., counts)
    plt.show()

if __name__ == "__main__":
    test_soft_nn_sim()  