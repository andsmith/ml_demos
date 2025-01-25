import numpy as np
import matplotlib.pyplot as plt
from util import get_n_disp_colors


def make_data(n):
    """
    Return an N x N matrix.
    """
    data = np.random.rand(n, n)
    return data


def _get_good_markersizes(sizes, img_size, density=0.8):
    """
    The goal is to have a certain fraction of the canvas colored by points.
    Make each cluster cover approximately the same area, so scale marker sizes
    by the square root of the number of points.   (i.e. assume no overlapping
    points, but plot them anyway.)  
        NOTE:  this is too big, return sqrt(area)

    :param sizes: list of integers, number of points in each cluster
    :param img_size: tuple of integers, w,h of pixels to cover
    :param density: float, fraction of the canvas to be covered by points
    :return: list of integers, marker sizes for each cluster
    """
    sizes = np.array(sizes)
    total_points = np.sum(sizes)
    n_clusters = len(sizes)

    # pixels squared per cluster
    px2_per_cluster = img_size[0] * img_size[1] * density / n_clusters
    area_per_marker = px2_per_cluster / sizes

    return np.sqrt(area_per_marker)


def test_get_good_markersizes():
    sizes = [10, 20]
    img_size = (100, 100)
    markersizes = _get_good_markersizes(sizes, img_size, density=0.5)
    print(markersizes)
    assert markersizes[0] == 5
    assert markersizes[1] == 7
    print("All tests passed!")


def add_alpha(colors, alpha):
    """
    Add an alpha channel to the colors.
    :param colors: M x 3 array of colors
    :param alpha: float in [0, 1]
    :return: M x 4 array of colors
    """
    return np.concatenate([colors, np.ones((colors.shape[0], 1)) * alpha], axis=1)


def plot_graph_stats(fig, ax, sim_matrix):
    """
    Plot the graph statistics, a histogram of values in the upper triangle of the similarity matrix.
    :param ax: matplotlib axis object
    :param sim_matrix: N x N similarity matrix
    """
    s = sim_matrix.shape[0]
    upper_triangle = sim_matrix[np.triu_indices(s, k=1)]
    n = len(upper_triangle)
    # check if binary
    if len(np.unique(upper_triangle)) == 2:
        n_bins = 2
        bin_centers = np.array((0.0, 1.0))
        count0 = np.sum(upper_triangle == 0)
        count1 = n - count0
        density = np.array([count0/n, count1/n])
        ax.bar(bin_centers, density, width=0.5)
        # ax.set_title("BINARY Graph has %i edge weights in %i bins" % (n, n_bins))
        # annotate both bars with the densities
        ax.text(0, .1, f'{density[0]:.5f}', ha='center', va='bottom')
        ax.text(1, .1, f'{density[1]:.5f}', ha='center', va='bottom')
    else:
        range = np.min(upper_triangle), np.max(upper_triangle)
        n_bins = np.min((s, 30))
        counts, bin_edges = np.histogram(upper_triangle, bins=n_bins, range=range)
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        # density = counts / n
        # ax.plot(bin_centers, density,'o-')

        ax.plot(bin_centers, counts, 'o-')
        ax.set_yscale('log')

    ax.set_title('Graph(v=%i) edge hist.' % s)
    fig.tight_layout(rect=[0, 0, 1, 1])


def plot_clustering(ax, points, colors, cluster_ids, image_size, alpha=.5, invert_y=True):
    """
    Plot the points colored by cluster.
    :param ax: matplotlib axis object
    :param points: N x 2 array of points
    :param colors: M x 3 array of colors
    :param cluster_ids: N array of integers in [0, M-1]
    """
    id_list = np.unique(cluster_ids)
    cluster_sizes = [np.sum(cluster_ids == i) for i in id_list]
    point_sizes = _get_good_markersizes(cluster_sizes, image_size)
    for i in id_list:
        cluster_points = points[cluster_ids == i]
        # print("Plotting cluster %i with %i points, markersize %i"%(i, len(cluster_points), point_sizes[i]))
        if invert_y:
            ax.scatter(cluster_points[:, 0], -cluster_points[:, 1], c=[colors[i]], s=point_sizes[i], alpha=alpha)
        else:
            ax.scatter(cluster_points[:, 0], cluster_points[:, 1], c=[colors[i]], s=point_sizes[i], alpha=alpha)
    ax.set_aspect('equal')
    ax.axis('off')


def plot_eigenvecs(fig, axes, vecs, n_max, k, colors=None, *args, **kwargs):
    """
    Plot the first n_max eigenvectors aranged vertically.
    Draw a red line between plots k and k+1.
    :param axes: list of n_max axes objects
    :param vecs: eigenvectors, N x N matrix
    :param n_max: number of eigenvectors to plot
    :param k: draw a line after this many plots 
    :param colors: dictionary with:
        'colors': list of M colors [r, g, b] in [0, 255]
        'ids': list of N integers in [0, M-1]
        If this is present, draw component j of each eigenvector in colors['colors'][colors['ids'][j]]
    """
    if axes is None or fig is None:
        fig, axes = plt.subplots(n_max, 1, figsize=(5, 5))

    fixed_colors = [c/255. for c in colors['colors']]

    comps_by_color = {i: np.where(colors['ids'] == i)[0] for i in range(len(colors['colors']))}

    for i in range(n_max):
        if colors is None:
            axes[i].plot(vecs[:, i])
        else:
            for c_i, color in enumerate(fixed_colors):
                # if points are ever out of order (i.e. not contiguous in color), this will look messed up, use 'o' or '.' instead then.
                axes[i].plot(comps_by_color[c_i], vecs[:, i][comps_by_color[c_i]], color=color, *args, **kwargs)

        axes[i].set_xticks([])
        axes[i].set_yticks([])
        axes[i].tick_params(axis='y', labelsize=6)
        axes[i].set_ylabel("%i" % i, fontsize=8)
        for pos in ['right', 'top', 'bottom', 'left']:
            axes[i].spines[pos].set_visible(False)
    fig.suptitle('Eigenvectors')
    fig.tight_layout(rect=[0, 0, 1, 1])

    # Draw lines between plots k-1 and k
    if k > 0 and k < n_max:
        above_bbox = axes[k-1].get_position()
        below_bbox = axes[k].get_position()
        line_y = (above_bbox.y1 + below_bbox.y0) / 2
        fig.add_artist(plt.Line2D((0, 1), (line_y, line_y), color='red', linewidth=1))


def test_plot_eigenvecs():
    from colors import COLORS
    data = make_data(100)
    colors = [COLORS['red']/255.,
              COLORS['green']/255.,
              COLORS['blue']/255.]
    ids = np.zeros(100, dtype=np.int32)
    ids[34:67] = 1
    ids[67:] = 2
    fig, axes = plt.subplots(8, 1, figsize=(5, 5))
    plot_eigenvecs(fig, axes, data, 8, 3, colors={'colors': colors, 'ids': ids})
    plt.show()


def test_plot_clustering():
    cluster_size_range = (2, 600)
    # cluster_sizes = [1, 2, 5, 10, 100, 500, 1000, 10000]

    def _n_clusters(n_clusters):
        points, ids = [], []
        for c_id in range(n_clusters):
            n = np.random.randint(*cluster_size_range)  # cluster_sizes[c_id]
            center = np.random.randn(2)*7
            sigma = np.random.rand(1)*6
            points.append(np.random.randn(n, 2)*sigma + center)
            ids.append(np.ones(n, dtype=np.int32)*c_id)
        colors = get_n_disp_colors(n_clusters) / 255.
        points = np.concatenate(points)
        ids = np.concatenate(ids)
        # print("Plotting points(%s), ids(%s), and colors(%s)." %(points.shape, ids.shape, colors.shape))

        fig_size = (5, 5)
        dpi = 100
        _, axes = plt.subplots(1, 1, figsize=(5, 5), dpi=dpi)
        img_size = (fig_size[0]*dpi, fig_size[1]*dpi)

        plot_clustering(axes, points, colors, ids, img_size)
        plt.title("Clustering")
        plt.tight_layout()
        plt.show()

    _n_clusters(10)
    _n_clusters(2)
    _n_clusters(1)


def test_graph_stats():
    from util import get_n_disp_colors
    from similarity import EpsilonSimGraph, FullSimGraph
    cluster_size_range = (2, 600)
    fig, axes = plt.subplots(3, 2)

    def _n_clusters(axes, n_clusters, graph_class, graph_kwargs={}):
        points, ids = [], []
        for c_id in range(n_clusters):
            n = np.random.randint(*cluster_size_range)  # cluster_sizes[c_id]
            center = np.random.randn(2)*7
            sigma = np.random.rand(1)*6
            points.append(np.random.randn(n, 2)*sigma + center)
            ids.append(np.ones(n, dtype=np.int32)*c_id)
        colors = get_n_disp_colors(n_clusters) / 255.
        points = np.concatenate(points)
        ids = np.concatenate(ids)
        axes[0].scatter(points[:, 0], points[:, 1], c=colors[ids], s=5)
        axes[0].set_title("%i clusters, %i points" % (n_clusters, len(points)))
        sim_graph = graph_class(points, **graph_kwargs)
        mat = sim_graph.get_matrix()
        plot_graph_stats(fig, axes[1], mat)

    # import ipdb; ipdb.set_trace()
    _n_clusters(axes[0], 10, EpsilonSimGraph, {'epsilon': 1.5})
    _n_clusters(axes[1], 2, EpsilonSimGraph, {'epsilon': 0.5})
    _n_clusters(axes[2], 1, FullSimGraph, {'sigma': 3.5})
    plt.tight_layout()
    plt.show()


def project_binary_clustering(points, labels, whiten=False):
    """
    :param points: N x d array of points
    :param labels: N array of integers in [0, 1], cluster assignments
    :return: 2D projection of points (N x 2 array) in the unit square
    """
    points = points - np.mean(points, axis=0)
    p0 = np.mean(points[labels == 0], axis=0)
    p1 = np.mean(points[labels == 1], axis=0)
    horizontal = p1 - p0
    horizontal /= np.linalg.norm(horizontal)
    points_deflated = points - np.dot(points, horizontal[:, np.newaxis]) * horizontal
    # find first principal component in the deflated space as the vertical axis
    covar = np.cov(points_deflated, rowvar=False)
    vals, vecs = np.linalg.eigh(covar)
    pc_dir0 = vecs[:, np.argmax(vals)]
    vert = pc_dir0 / np.linalg.norm(pc_dir0)

    # check = np.abs(np.dot(horizontal, vert))
    # print("Check orthogonality: %f" % check)

    proj_mat = np.column_stack((horizontal, vert))
    projected = points @ proj_mat
    if whiten:
        projected = (projected - np.mean(projected, axis=0)) / np.std(projected, axis=0)

    # scale to unit square
    unit = (projected - np.min(projected, axis=0)) / (np.max(projected, axis=0) - np.min(projected, axis=0))

    return unit


def plot_binary_clustering(ax, points, labels, true_labels=None, point_size=5, circle_size=50):
    """
    (Will flip predicted labels if more than half are wrong)
    :param ax: matplotlib axis object
    :param points: N x 2 array of points
    :param labels: N array of integers in [0, 1], cluster assignments
    :param true_labels: N array of integers in [0, 1], ground truth labels
    """
    labels =np.array(labels,dtype=np.int32)
    colors = np.array([(0.122, 0.467, 0.706), # matplotlib blue
                       (1.0, 0.498, 0.055)])
    ax.scatter(points[:, 0], points[:, 1], c=colors[labels], s=point_size)

    if true_labels is None:
        return

    true_labels = np.array(true_labels,dtype=np.int32)
    error = labels != true_labels
    if np.mean(error)>.5:
        labels = 1 - labels
        error = labels != true_labels
    # circles around errors, edge color of correct label (no face color)
    correct_colors = colors[true_labels[error]]
    ax.scatter(points[error, 0], points[error, 1], s=circle_size, edgecolors=correct_colors,facecolors='none')



def test_project_binary_clustering():
    n, d = 300, 10
    points = np.vstack((np.random.randn(n, d) + np.random.randn(d)*1.0,
                        np.random.randn(n, d) + np.random.randn(d)*1.0))
    true_labels = np.concatenate((np.zeros(n), np.ones(n))).astype(np.int32)
    error = np.random.rand(n*2) < 0.05
    pred_labels = true_labels.copy()
    pred_labels[error] = 1 - pred_labels[error]
    points_flattened = project_binary_clustering(points, pred_labels, whiten=True)
    fig,ax = plt.subplots(1, 1)
    plot_binary_clustering(ax, points_flattened, pred_labels, true_labels)
    plt.title("Projected points,\nerrors circled w/correct colors.")
    plt.show()

if __name__ == "__main__":
    # test_plot_eigenvecs()
    # test_plot_clustering()
    # test_graph_stats()
    test_project_binary_clustering()
    print("All tests passed!")
