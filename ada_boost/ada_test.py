"""
Sandbox to run scikit.learn AdaBoostClassifier
"""
from classify import plot_classifier
from spiral import make_minimal_data
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import logging

def ada_test():
    X, y = make_minimal_data()
    clf = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=8, algorithm='SAMME')

    clf.fit(X, y)
    y_pred = clf.predict(X)
    print("Iterations(estimators):", clf.n_estimators) # number of estimators actually trained
    print('Accuracy:', accuracy_score(y, y_pred))
    print("Frac correct:", np.mean(y == y_pred))
    print('Predictions:', y_pred)
    print('Alphas:', clf.estimator_weights_)
    print('Errors:', clf.estimator_errors_)
    fix, ax = plt.subplots()
    plot_classifier(ax, X, y, model=clf, boundary=None, markersize=18, bubble_size=250)
    
    for i, est in enumerate(clf.estimators_):
        # print axis and threshold of decision stump
        print('Estimator', i, 'axis:', est.tree_.feature[0], 'threshold:', est.tree_.threshold[0])
        # draw the line and label it with the iteration number
        if est.tree_.feature[0] == 1:
            plt.plot(ax.get_xlim(), [est.tree_.threshold[0], est.tree_.threshold[0]], 'k--') 
            plt.text(ax.get_xlim()[1], est.tree_.threshold[0], str(i), verticalalignment='top')
        else:
            plt.plot([est.tree_.threshold[0], est.tree_.threshold[0]], ax.get_ylim(), 'k--')
            plt.text(est.tree_.threshold[0], ax.get_ylim()[1], str(i), verticalalignment='top')

    plt.axis('equal')
    plt.legend(['classified 0', 'classified 1', 'misclassified 1', 'misclassified 0','decision boundary $n$'])
    plt.title("Final classification of AdaBoost with (%i) decision stumps\nAccuracy: %.2f" % (clf.n_estimators, accuracy_score(y, y_pred))) 
    plt.show()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    ada_test()
