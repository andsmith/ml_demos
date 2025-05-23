import numpy as np
from sklearn.cluster import KMeans
from abc import ABC, abstractmethod
from util import get_good_point_size

class ClusteringAlgorithm(ABC):
    """
    Abstract class for clustering algorithms, plugins to the cluster creator.
    """

    def __init__(self, name, k):
        self._name = name
        self._k = k
        self._fit = False

    @abstractmethod
    def fit(self, x):
        """
        Cluster the points.
        :param x: N x 2 array of points
        :returns: N x 1 array of cluster assignments
        """
        pass

    @abstractmethod
    def assign(self, x):
        """
        Assign clusters to new points
        :param x: N x 2 array of points
        :returns: N x 1 array of cluster assignments
        """
        pass
    def is_fit(self):
        return self._fit    
    
    def get_k(self):
        return self._k
    
def render_clustering(img, points, cluster_ids, colors, clip_unit=True, margin_px=5):
    """
    Render the clustering.
    :param img: the image to render on
    :param points: N x 2 array of points
    :param cluster_ids: N x 1 array of cluster assignments
    :param colors: list of colors for each cluster
    :param clip_unit: if True, draws only points in unit square, else draws all points scaled to img size
    :param margin_px: don't put points in the margin
    :param pt_size: size of points (pt_size x pt_size boxes)
    """
    pt_size = get_good_point_size(points.shape[0], None)
    points_scaled = (points * img.shape[1::-1]).astype(int)
    if cluster_ids is None:
        cluster_ids = np.zeros(points.shape[0], dtype=np.int32)

    if clip_unit:
        valid = (points_scaled[:, 0] >= margin_px) & (points_scaled[:, 0] < img.shape[1] - margin_px) & \
                (points_scaled[:, 1] >= margin_px) & (points_scaled[:, 1] < img.shape[0] - margin_px)
        points = points[valid]
        cluster_ids = cluster_ids[valid]
        points_scaled = points_scaled[valid]

    if isinstance(colors, np.ndarray):
        colors = colors.tolist()
    for i, (x, y) in enumerate(points_scaled):
        color = colors[cluster_ids[i]]

        img[y:y + pt_size, x:x + pt_size] = color


class KMeansAlgorithm(ClusteringAlgorithm):
    def __init__(self, k):
        super().__init__('KMeans', k)
        self._kmeans = KMeans(n_clusters=k)

    def fit(self, x):
        self._kmeans.fit(x)
        self._fit = True
        return self._kmeans.labels_
    
    def assign(self, x):
        if not self._fit:
            raise ValueError("Model has not been fit yet.")
        return self._kmeans.predict(x)


class SpectralAlgorithm(ClusteringAlgorithm):
    def __init__(self, sim_graph, normalize=False):
        """
        :param sim_graph: similarity graph
        """
        super().__init__('Spectral', None)
        self._normalize = normalize
        self._g = sim_graph
        self._tree = self._g.get_tree()  # for clustering new points
        self._solve()
        self._kmeans = None

    def _solve(self):
        w = self._g.get_matrix().copy()
        # set diagonal to zero
        np.fill_diagonal(w, 0)
        # compute the Laplacian matrix:
        degree_vec = np.sum(w, axis=1)
        degree_mat = np.diag(degree_vec)
        laplacian = degree_mat - w
        if self._normalize:
            # normalize
            degree_mat_sqrt = np.diag(1 / np.sqrt(degree_vec))
            laplacian = np.dot(degree_mat_sqrt, np.dot(laplacian, degree_mat_sqrt))

        eigvals, eigvecs = np.linalg.eigh(laplacian)

        # sort by eigenvalues
        idx = eigvals.argsort()
        self._eigvals = eigvals[idx]
        self._eigvecs = eigvecs[:, idx]


    def fit(self, n_clusters, n_features=None):
        n_features = n_features if n_features is not None else n_clusters
        eig_features = self._eigvecs[:, :n_features]
        if self._normalize:
            # normalize
            eig_features /= np.linalg.norm(eig_features, axis=1)[:, np.newaxis]

        # kmeans on eigenvectors
        self._kmeans = KMeans(n_clusters=n_clusters)
        self._kmeans.fit(eig_features)

        # cluster
        cluster_ids = self._kmeans.labels_

        return cluster_ids

    def get_eigens(self):
        return self._eigvals, self._eigvecs 
    
    def assign(self, x):
        if self._kmeans is None:
            raise ValueError("Model has not been fit() yet.")
        # get index of nearest neighbor to x
        n_ind = self._tree.query(x, k=1)[1]
        return self._kmeans.labels_[n_ind]


def test_render_clustering():
    import cv2
    img = np.zeros((480, 640, 3), np.uint8)
    points = np.random.rand(100, 2)
    cluster_ids = np.random.randint(0, 3, 100)
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    render_clustering(img, points, cluster_ids, colors)
    cv2.imshow('test_render_clustering', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    test_render_clustering()
