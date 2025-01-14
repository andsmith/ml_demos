from clustering import ClusteringAlgorithm
import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors
import cv2
from enum import IntEnum
from util import image_from_floats

class SimilarityGraphTypes(IntEnum):
    EPSILON = 0
    NN = 1
    FULL = 2


SIMGRAPH_PARAM_NAMES = {SimilarityGraphTypes.NN: "N-nearest",
                        SimilarityGraphTypes.EPSILON: "Epsilon",
                        SimilarityGraphTypes.FULL: "Sigma"}

SIMGRAPH_KIND_NAMES = {SimilarityGraphTypes.NN: "N-nearest",
                        SimilarityGraphTypes.EPSILON: "Epsilon",
                        SimilarityGraphTypes.FULL: "Full"}

def get_kind_from_name(names, name):
    for kind, kind_name in names.items():
        if kind_name == name:
            return kind
    raise ValueError(f"Invalid name: {name}")

class SimilarityGraph(object):
    """
    Build a similarity graph from a set of points using euclidean distances.
    """

    def __init__(self, points, kind, epsilon_dist=.01, n_nearest=10):
        """
        Construct a similarity graph from points in the unit square.
        :param points: 2D numpy array of points
        :param kind: type of similarity graph to build, one SimilarityGraphTypes.
        :param epsilon_dist: epsilon distance for epsilon graph
            M[i,j] = 1 if dist(i,j) < epsilon, 0 otherwise
        :param n_nearest: number of nearest neighbors for K-NN graph
            M[i,j] = 1 if j is among the n_nearest neighbors of i, 0 otherwise


        """
        self._points = points
        self._kind = kind

        if kind == SIMGRAPH_KIND_NAMES[SimilarityGraphTypes.EPSILON]:
            self._mat, self._img = self._build_epsilon_sim_matrix(epsilon_dist)
        elif self._kind == SIMGRAPH_KIND_NAMES[SimilarityGraphTypes.NN]:
            self._mat, self._img = self._build_knn_sim_matrix(n_nearest)
        elif self._kind == SIMGRAPH_KIND_NAMES[SimilarityGraphTypes.FULL]:
            self._mat, self._img = self._build_full_sim_matrix(np.inf)
        else:
            raise ValueError(f"Invalid kind: {self._kind}")

        #print("Built similarity matrix, shape %s, mean %.3f" % (self._mat.shape,self._mat.mean()))

    def _build_epsilon_sim_matrix(self, max_dist):
        """
        Build the similarity matrix using epsilon distance.
        I.e., two points are connected if their distance is less than epsilon, all 
        weight 1 or 0.
        """
        #print("Recomputing similarity matrix with max_dist: %.3f" % max_dist)
        dists = squareform(pdist(self._points))
        sim_matrix = np.zeros(dists.shape)
        sim_matrix[dists < max_dist] = 1
        img = image_from_floats(sim_matrix, 0, 1)
        img = cv2.merge([img, img, img])
        return sim_matrix , img

    def _build_full_sim_matrix(self, max_dist):
        raise NotImplementedError("Full graph not implemented")

    def get_matrix(self):
        return self._mat
    
    def get_image(self):
        return self._img

    def _build_knn_sim_matrix(self, k, mutual=False):
        """
        Build the similarity matrix using K-nearest neighbors, i.e.
        two points are connected if they are among each other's K-nearest neighbors.
        """
        raise NotImplementedError("KNN method not finished!")
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree').fit(self._points)
        distances, indices = nbrs.kneighbors(self._points)
        n_points = self._points.shape[0]
        sim_matrix = np.zeros((n_points, n_points))
        for i in range(n_points):
            # add it's neighbors and their distances to the similarity matrix
            for j, dist in zip(indices[i], distances[i]):
                sim_matrix[i, j] = np.exp(-dist**2 / (2 * dist**2))
        # make it symmetric
        if not mutual:
            raise NotImplementedError("KNN not implemented")
            # knn graph
            sim_matrix = np.maximum(sim_matrix, sim_matrix.T)
        else:
            raise NotImplementedError("Mutual KNN not implemented")
            # mutual KNN graph
            mask = sim_matrix > 0
            sim_matrix = np.minimum(sim_matrix, sim_matrix.T)
            sim_matrix = sim_matrix * mask

        return sim_matrix

    def draw_graph(self,img):
        """
        Draw an edge between points with nonzero entries in the similarity matrix.
        """
        lines = []
        nonzero = np.nonzero(np.triu(self._mat))
        for i, j in zip(*nonzero):
            lines.append(np.array([self._points[i], self._points[j]], dtype=np.int32))

        cv2.polylines(img, lines, False, (128, 128, 128), 1, cv2.LINE_AA)
