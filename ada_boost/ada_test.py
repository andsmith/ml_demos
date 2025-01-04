"""
Sandbox to run scikit.learn AdaBoostClassifier
"""
from classify import plot_classifier
from make_data import make_minimal_data, make_spiral_data, make_checker_data, make_bump
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import logging
import sys


def plot_boundary(ax, X, y, clf):
    n_pts = 1000
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    x = np.linspace(x_min, x_max, n_pts)
    y = np.linspace(y_min, y_max, n_pts)
    xx, yy = np.meshgrid(x, y)
    points = np.vstack([xx.ravel(), yy.ravel()]).T
    preds = clf.predict(points)
    print(np.unique(preds))
    preds = preds.reshape(n_pts, n_pts)
    # ax.contourf(xx, yy, preds, cmap=plt.cm.RdBu, alpha=0.8)
    # show predictions as an image, with the color indicating the prediction
    image = ax.imshow(preds, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap='RdBu', alpha=.6)

    # plt.colorbar(mappable=image, cax=None, ax=ax)
    image.set_zorder(-1)
    ax.set_aspect('equal')


def ada_test(X, y, n_estimators=100):
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                             n_estimators=n_estimators, algorithm='SAMME')

    clf.fit(X, y)
    y_pred = clf.predict(X)
    print("Iterations(estimators):", clf.n_estimators)  # number of estimators actually trained
    print('Accuracy:', accuracy_score(y, y_pred))
    print("Frac correct:", np.mean(y == y_pred))
    
    _, ax = plt.subplots()

    plot_classifier(ax, X, y, model=clf, boundary=None)
    plot_boundary(ax, X, y, clf)
    x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
    print(x_lim, y_lim)
    for i, est in enumerate(clf.estimators_):
        # draw the line and label it with the iteration number
        if est.tree_.feature[0] == 1:
            plt.plot(ax.get_xlim(), [est.tree_.threshold[0], est.tree_.threshold[0]], 'k--')
            plt.text(ax.get_xlim()[1], est.tree_.threshold[0], str(i), verticalalignment='top')
        else:
            plt.plot([est.tree_.threshold[0], est.tree_.threshold[0]], ax.get_ylim(), 'k--')
            plt.text(est.tree_.threshold[0], ax.get_ylim()[1], str(i), verticalalignment='top')

    print(x_lim, y_lim)
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    # plt.axis('equal')
    plt.legend(['class 0 hit', 'class 1 hit', 'class 1 miss', 'class 0 miss', 'stump $n$'] ,loc='lower right')
    plt.title("Final classification of AdaBoost with (%i) decision stumps\nAccuracy: %.2f" %
              (clf.n_estimators, accuracy_score(y, y_pred)))
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    logging.info("AdaBoost test:  Run sklearn AdaBoostClassifier on a dataset, plot resulting decision boundary and correct/incorrect classifications") 

    if '-p' in sys.argv:
        n_points = int(sys.argv[sys.argv.index('-p')+1])
    else:
        n_points = 50
        
    if 'spiral' in sys.argv:
        X, y, _ = make_spiral_data(n_points, turns=1.0, ecc=1.0, margin=0.04, random=False)
    elif 'minimal' in sys.argv:
        X, y = make_minimal_data()
    elif 'checker' in sys.argv:
        X, y = make_checker_data(n_points,1)
    elif 'bump' in sys.argv:
        X, y = make_bump(n_points, 0.15, 0.2, 0.0, noise_frac=0.00, separable = True)
    else:
        raise ValueError("Please specify a dataset: spiral, minimal, checker, bump")

    if '-n' in sys.argv:
        n_estimators = int(sys.argv[sys.argv.index('-n')+1])
    else:
        n_estimators = 100

    if "?" in sys.argv:
        print("Usage: python ada_test.py <dataset> [-n <n_estimators>] [-p] [?]")
        print("  dataset: spiral, minimal, checker, bump")
        print("  n_estimators: number of decision stumps to train")
        print("  -p: number of points (grid side length) for datasets (except minimal)")
        print("  -n: number of decision stumps to train")
        print("  ?: print this message")
        sys.exit(0)
    ada_test(X, y, n_estimators=n_estimators)
