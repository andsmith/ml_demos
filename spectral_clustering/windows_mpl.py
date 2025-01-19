"""
Class for output windows, using matplotlib.
"""

import numpy as np
import matplotlib.pyplot as plt
from util import get_n_disp_colors
from layout import WINDOW_LAYOUT


class ValuesWindow(object):
    """
    Class for displaying:
        * Spectrum,
        * random projections,
        * Similarity matrix, and
        * Edge weight statistics
        """

    def __init__(self, app, n_to_plot, f_to_use, init_noise=0):
        self._app = app
        self._sim_graph = None
        self._plot_params = {'f': f_to_use,
                             'n': n_to_plot,
                             'noise': init_noise}
        self._setup_figs()

    def _setup_figs(self):
        self._subplot_inds = WINDOW_LAYOUT['plots']
        n_rows = np.max([v for v in self._subplot_inds.values()])  # max number of rows needed
        n_cols = len(set(v for v in self._subplot_inds.values()))
        self._fig, self._axes = plt.subplots(n_rows, n_cols)

    def _update_features(self):
        self._features = self._e_vects[:, :self._plot_params['f']]

    def update_sim_graph(self, sim_graph, eigenvalues, eigenvectors):
        self._sim_graph = sim_graph
        self._e_vals= eigenvalues
        self._e_vecs = eigenvectors
        self._update_features()
        self._refresh()

    def update_plot_param(self, param, val):
        if param not in self._plot_params:
            raise ValueError("Invalid parameter: %s" % param)
        self._plot_params[param] = val
        
        self._refresh()

    def _refresh(self):
        self._plot_spectrum( self._axes[self._subplot_inds['spectrum']])
        self._plot_randproj(self._fig,self._axes[self._subplot_inds['rand_proj']])
        self._plot_sim_matrix(self._axes[self._subplot_inds['sim_matrix']])
        self._plot_weight_stats(self._axes[self._subplot_inds['weight_stats']])

        self._fig.tight_layout(rect=[0, 0, 1, 1])

    def _plot_spectrum(self, ax):
        if self._sim_graph is None:
            return
        n_features = self._app.toolbar.get_value('f')
        if self._n_to_plot < n_features:
            self._n_to_plot = n_features
        ax.plot(self._e_vals[:self._n_to_plot], 'o-')
        # draw a vertical red line at f
        x_k = n_features - 0.5
        ax.axvline(x=x_k, color='r', linestyle='-')
        ax.set_title("Spectrum")

    def _plot_sim_matrix(self, ax):
        if self._sim_graph is None:
            return

        img = self._sim_graph.get_sim_matrix()
        ax.imshow(img, cmap='viridis', interpolation='nearest')
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title("Similarity matrix")

    def _plot_weight_stats(self, ax):
        if self._sim_graph is None:
            return
        weights = self._sim_graph.get_matrix().reshape(-1)
        ax.hist(weights, bins=100)
        ax.set_title("Edge weight statistics")


    def _remake_axes(self):
        if self._features is None:
            return
        # make random axes, orthogonal in feature space
        self._axes = np.random.randn(2, self._features.shape[1])
        # normalize lengths
        self._axes[0] /= np.linalg.norm(self._axes[0])
        self._axes[1] -= self._axes[1] @ self._axes[0] * self._axes[0]
        self._axes[1] /= np.linalg.norm(self._axes[1])
        # check orthogonality
        err = np.abs(np.sum(self._axes[0] * self._axes[1]))
        if err > 1e-6:
            raise ValueError("Axes are not orthogonal!?")
        self.refresh()

    def _plot_randproj(self, fig, ax):
        if self._sim_graph is None:
            return
        points = self._features @ self._axes.T
        noisy_points = points + self._noise_offsets * self._noise * .1
        colors = self.app.ui_window.get_cluster_color_ids()
        # import ipdb; ipdb.set_trace()

        plot_clustering(ax, noisy_points, colors['colors']/255., colors['ids'], image_size=self._bbox_size, alpha=0.5)
        # since clusters will be on the border if there are many connected components, move everything in by a percentage
        marg_frac = 0.025
        x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
        w, h = x_lim[1] - x_lim[0], y_lim[1] - y_lim[0]
        ax.set_xlim(x_lim[0] - marg_frac*w, x_lim[1] + marg_frac*w)
        ax.set_ylim(y_lim[0] - marg_frac*h, y_lim[1] + marg_frac*h)
        ax.set_title("Random Projection")
        self._disp_img = self._plotter.render_fig(fig)




class ResultsWindow(object):
    pass
class EigenvectorsWindow(object):
    
    def _plot_eigenvectors(self, fig, ax):
        if self._sim_graph is None:
            return
        plot_eigenvecs(fig, ax, self._e_vecs, n_max=self._n_to_plot,
                       k=self._f_to_use, colors=self._app.ui_window.get_cluster_color_ids())


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
        print("Plotting points(%s), ids(%s), and colors(%s)." % (points.shape, ids.shape, colors.shape))

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


class FakeToolbar(object):
    def get_value(self, which):
        if which == 'f':
            return 10
        elif which == 'k':
            return 4


class FakeUiWindow(object):
    def __init__(self, n):
        self._n = n

    def get_cluster_color_ids(self):
        ids = np.ones(self._n, dtype=np.int32)
        colors = np.array([[128, 255, 255], ], dtype=np.uint8)
        return {'ids': ids, 'colors': colors}


class FakeApp(object):
    def __init__(self, pts):
        self.toolbar = FakeToolbar()
        self.ui_window = FakeUiWindow(pts.shape[0])

def test_values_window():

    from similarity import FullSimGraph
    from clustering import SpectralAlgorithm
    # test_plot_eigenvecs()
    # test_plot_clustering()
    n = 100
    points = np.random.randn(n*2, 2)
    sg = FullSimGraph(points, sigma=0.025)
    app = FakeApp(points)
    sa = SpectralAlgorithm(sg)
    

    window = ValuesWindow(app, 10, 3)
    e_vals, e_vecs = sa.get_eigens()
    window.update_sim_graph(sg,e_vals, e_vecs)
    plt.show()


if __name__ == "__main__":
    test_values_window()
    print("All tests passed!")
