import numpy as np
import matplotlib.pyplot as plt
import cv2


def make_line_data(n_pts, n_outliers, noise_sigma=0.01):
    # make 2d points in a line, add noise & outlier points
    # make sure they are sufficiently far away, so the line is long enough for
    # the demo to be visualized clearly.
    min_sep = 0.6
    while True:
        p0 = np.random.rand(2).reshape(-1, 1)
        p1 = np.random.rand(2).reshape(-1, 1)
        if np.linalg.norm(p1 - p0) > min_sep:
            break
    t = np.random.rand(n_pts)
    line_pts = np.tile(p0, (1, n_pts)) * t + np.tile(p1, (1, n_pts)) * (1. - t)
    line_pts += np.random.randn(n_pts*2).reshape(2, -1)*noise_sigma
    error_pts = np.random.rand(n_outliers*2).reshape(-1, 2)
    params = fit_line(line_pts.T)
    return np.vstack((line_pts.T, error_pts)), params


def fit_line(data):
    """
    Fit a line to the set of points
    :param data: a Nx2 array of points
    :returns: the parameters a, b, c of the line a*x + b*y + c = 0
    """
    x, y = data[:, 0], data[:, 1]
    A = np.vstack((x, y, np.ones_like(x))).T
    return np.linalg.svd(A)[2][-1]


def point_line_distances(pts, a, b, c):
    """
    Return the perpendicular distance between points pts (Nx2) and the line defined
    by a*x + b*y + c = 0.
    """
    n = np.sqrt(a**2 + b**2)
    return np.abs(a*pts[:, 0] + b*pts[:, 1] + c) / n


def plot_line(a, b, c, ax=None, plt_args=(), plt_kwargs={}):
    """
    Add a line to the current plot with parameters a,b,c within current axis limits.
    """
    ax = ax if ax is not None else plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if np.isclose(b, 0):
        # vertical line, points are where y=0 and y=1
        x = [-c/a, -c/a]
        y = [ylim[0], ylim[1]]
    else:
        # otherwise, find points where x=0 and x=1.
        x = [xlim[0], xlim[1]]
        y = [-(c+a*x[0])/b, -(c+a*x[1])/b]
    ax.plot(x, y, *plt_args, **plt_kwargs)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    return ax


def test_fit_line():
    data, params = make_line_data(100, 0)
    a, b, c = fit_line(data)
    assert np.allclose([a, b, c], params) or np.allclose([a, b, c], -params), 'Line fit failed.'


def test_point_line_distances():
    a, b, c = 1, 1, -1
    pts = np.array([[0, 0], [1, 1], [1, 0], [0, 1]])
    true_dists = [np.sqrt(2)/2, np.sqrt(2)/2, 0, 0]
    dists = point_line_distances(pts, a, b, c)
    assert np.allclose(dists, true_dists), 'Point line distances failed: %s != %s' % (dists, true_dists)


if __name__ == '__main__':
    test_fit_line()
    test_point_line_distances()
    print('All tests passed.')
