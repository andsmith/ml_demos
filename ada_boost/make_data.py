"""
Create a classification dataset with a spiral decision boundary.
"""
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from plotting import plot_dataset, plot_classifier, LABEL_COLORS
import logging


def make_spiral_data(n_points, turns=.75, ecc=1.0, margin=0.04, random=False):
    """
    Create two classes separated by a spiraling
    decision boundary.  
    :param n_points: sqrt(number of points to generate) (i.e. side length if random=False)
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
    if random:
        points = np.random.rand(n_points**2, 2)*2 - 1
    else:
        n_points = np.ceil((n_points)).astype(int)
        points = np.linspace(-1, 1, n_points+2)[1:-1]
        points = np.vstack(np.meshgrid(points, points)).reshape(2, -1).T

    test_points = [Point(p) for p in points]
    labels = np.array([poly.contains(p) for p in test_points])
    labels = 2 * labels - 1
    return points, labels, square_boundary


def make_bump(n_points, h=.1, w=.1, x_left=0.0, noise_frac=0.000, random=False, separable=True):
    """
    Create a dataset separable by a horizontal line,except for a rectangular bump at the specified location.
    :param n_points: number of points on one side of the grid (in the unit square) to generate (if random = false, else total number of points)
    :param h: height of the bump
    :param w: width of the bump
    :param x_left: x location of the bump
    :param noise_frac: fraction of points to flip the label
    :param random: if True, generate random points, otherwise a grid
    """
    logging.info("Making bump classification dataset with %i points." % n_points)
    if random:
        points = np.random.rand(n_points, 2)
        dy = np.sqrt(n_points)
    else:
        n_points = np.ceil((n_points)).astype(int)
        points = np.linspace(0, 1, n_points+2)[1:-1]
        dy = points[1]-points[0]
        points = np.vstack(np.meshgrid(points, points)).reshape(2, -1).T
    labels = np.zeros(points.shape[0], dtype=bool)

    # set upper half of all points to True
    labels[points[:, 1] > .5] = True

    # set points in upper half and also in the bump to False
    labels[(points[:, 1] > .5) & (points[:, 0] > x_left) & (points[:, 0] < x_left + w) & (points[:, 1] < .5+h)] = False

    # flip some labels

    n_flip = int(noise_frac * points.size)
    print(n_flip, noise_frac)
    logging.info("Flipping %i of %i labels." % (n_flip, points.shape[0]))
    flip = np.random.choice(points.shape[0], n_flip)
    labels[flip] = ~labels[flip]

    # move positive samples down by dy * 1.5 so classes overlap
    if not separable:
        points[labels, 1] -= dy * 1.5

    # make labels in {-1, 1}
    labels = 2 * labels - 1

    return points, labels
def make_checker_data(n_pts=20, clip_cols=0):
    
    X0, X1 = np.meshgrid(np.linspace(-1, 1,n_pts), np.linspace(-1,1,n_pts))
    Y_in = (X0 > 0 ) ^ (X1 > 0)
    if clip_cols>0:
        X = np.hstack([X0[:,:-clip_cols].reshape(-1,1),X1[:,:-clip_cols].reshape(-1,1)])
        y = np.ones(X.shape[0])
        y[~Y_in[:,:-clip_cols].reshape(-1)] = -1
    else:
        X = np.hstack([X0.reshape(-1,1),X1.reshape(-1,1)])
        y = np.ones(X.shape[0])
        y[~Y_in.reshape(-1)] = -1

    return X, y

def make_minimal_data():
    """
    Create a dataset with two classes, arranged like:
       +  +
     -    
             +
       -  - 
    I.e. so an axis-aligned, linear decision boundary can only achieve 5/6 accuracy.
    :returns: points (xy), labels (-1,1)
    """
    points = np.array([[0, 0], # -
                       [1, 0],# -
                       #[2, 1.25], # +
                       #[-1, 1.75], # -
                       [1, 3], # +
                       [0, 3]]) # +
    labels = np.array([-1, 1, -1, 1, ], dtype=np.float64)
    return points, labels

def make_xor_data():
    points = np.array([[0, 0], # -)
                          [0, 1], # +
                          [1, 0], # +
                          [1, 1]]) # -
    labels = np.array([-1, 1, 1, -1], dtype=np.float64)
    return points, labels
    

def test_minimal():
    points, labels = make_minimal_data()
    plt.plot(points[labels==-1, 0], points[labels==-1, 1], '.',color=LABEL_COLORS[-1], label='class -1', markersize=10)
    plt.plot(points[labels==1, 0], points[labels==1, 1], '.',color=LABEL_COLORS[1], label='class 1', markersize=10)
    plt.legend()
    plt.show()

    from perceptron import DecisionStump
    clf = DecisionStump()
    clf.fit(points, labels)
    plot_classifier(plt.gca(), points, labels, clf, boundary=None)
    plt.show()
    

def test_bump():
    points, labels = make_bump(200)
    # plt.style.use('dark_background')
    plt.plot(points[labels, 0], points[labels, 1], '.', label='class 0', markersize=2)
    plt.plot(points[~labels, 0], points[~labels, 1], '.', label='class 1', markersize=2)
    plt.legend()
    plt.show()


def test_spiral():
    X, y, line = make_spiral_data(40000, turns=5, random=True)
    plt.style.use('dark_background')
    c1 = np.where(y == 0)[0]
    plt.plot(X[y, 0], X[y, 1], '.r', label='class 0', markersize=2, alpha=.3)
    plt.plot(X[~y, 0], X[~y, 1], '.b', label='class 1', markersize=2, alpha=.3)
    line = np.vstack([line, line[0]])
    plt.plot(line[:, 0], line[:, 1], 'w-', label='decision boundary')
    plt.legend()
    plt.show()
def test_checker():
    n_pts=20
    X, y = make_checker_data(n_pts=n_pts, clip_cols=0)
    plt.subplot(1,2,1)
    plt.plot(X[y==1, 0], X[y==1, 1], '.', label='class 1', markersize=2)
    plt.plot(X[y==-1, 0], X[y==-1, 1], '.', label='class -1', markersize=2)
    plt.title('Checkerboard dataset, %s x %s points' % (n_pts, n_pts))
    X, y = make_checker_data(n_pts=n_pts, clip_cols=1)
    plt.subplot(1,2,2)
    plt.plot(X[y==1, 0], X[y==1, 1], '.', label='class 1', markersize=2)
    plt.plot(X[y==-1, 0], X[y==-1, 1], '.', label='class -1', markersize=2)
    plt.title('Checkerboard dataset, %s x %s points' % (n_pts, n_pts-1))

    plt.show()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    #test_bump()
    test_minimal()
    test_checker()
