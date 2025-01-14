from clustering import ClusteringAlgorithm
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
import cv2
from enum import IntEnum
from util import image_from_floats
from abc import ABC, abstractmethod


class SimilarityGraphTypes(IntEnum):
    EPSILON = 0
    NN = 1
    FULL = 2


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
        """
        self._points = points
        self._mat, self._img = self._build()

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

    def get_image(self):
        return self._img


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
        sim_matrix = np.zeros(dists.shape)
        sim_matrix[dists < self._epsilon] = 1
        img = image_from_floats(sim_matrix, 0, 1)
        img = cv2.merge([img, img, img])
        return sim_matrix, img


class FullSimGraph(SimilarityGraph):
    def __init__(self, points, sigma):
        self._sigma = sigma
        super().__init__(points)

    def _build(self):
        raise NotImplementedError("Full similarity graph not implemented yet.")


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

    def _build(self ):
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
        img = image_from_floats(edge_mat, 0, 1)
        img = cv2.merge([img, img, img])
        return edge_mat, img


# labels for slider param for different simgraph types
SIMGRAPH_PARAM_NAMES = {SimilarityGraphTypes.NN: "N-nearest",
                        SimilarityGraphTypes.EPSILON: "Epsilon",
                        SimilarityGraphTypes.FULL: "Sigma"}
# simgraph type menu options
SIMGRAPH_KIND_NAMES = {SimilarityGraphTypes.NN: "N-nearest",
                       SimilarityGraphTypes.EPSILON: "Epsilon",
                       SimilarityGraphTypes.FULL: "Full"}
