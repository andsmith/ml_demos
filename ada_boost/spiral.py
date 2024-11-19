"""
Create a classification dataset with a spiral decision boundary.
"""
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
import logging

def make_spiral_data(n_points, turns=2.0, ecc=1.0, margin=0.04):
    """
    Create two classes separated by a spiraling
    decision boundary.  
    :param n_points: total number of points to generate
    :param turns: how many times the boundary is twisted
    :param ecc: squish or stretch the spiral
    :returns: points, labels
    """
    logging.info("Making spiral classification dataset with %i points and %.2f turns." % (n_points, turns))
    n_samples = int(100*turns*2)

    # make the boundary flat
    boundary = np.vstack([np.linspace(margin-1, 1-margin, n_samples),
                          np.zeros(n_samples)]).T

    # and twist it
    # thetas change slowly at [0] and [-1]
    n_margin = int(n_samples/2 * margin)
    thetas = ((np.cos(np.linspace(- np.pi, 0, n_samples//2)) + 1)/2)**2 * 2*np.pi * turns

    for i, theta in enumerate(thetas):
        r_mat = np.array([[np.cos(theta)*ecc, -np.sin(theta)*ecc],
                          [np.sin(theta), np.cos(theta)]])
        # rotate points i, n_samples-i
        boundary[i] = np.dot(r_mat, boundary[i])
        boundary[n_samples-i-1] = np.dot(r_mat, boundary[n_samples-i-1])
    square_boundary = np.vstack([(-1, -1),
                                 (-1.0, boundary[0, 1]),
                                 boundary,
                                 (1.0, boundary[-1, 1]),
                                 (1, -1)])
    poly = Polygon(square_boundary)
    points = np.random.rand(n_points, 2)*2 - 1
    test_points = [Point(p) for p in points]
    labels = np.array([poly.contains(p) for p in test_points])
    return points, labels, square_boundary


def test_spiral():
    X, y,line = make_spiral_data(4000, turns=2)
    plt.style.use('dark_background')
    c1 = np.where(y == 0)[0]
    plt.plot(X[y, 0], X[y, 1], 'r.', label='class 0', markersize=2)
    plt.plot(X[~y, 0], X[~y, 1], 'b.', label='class 1', markersize=2)
    plt.plot(line[:, 0], line[:, 1], 'w-', label='decision boundary')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    test_spiral()
