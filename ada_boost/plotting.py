import numpy as np
import matplotlib.pyplot as plt
LABEL_COLORS = {-1: 'red', 1: 'blue'}

POINT_MARKER_SIZE = 10
CIRCLE_SIZE = 100

def plot_dataset(ax, x, y, *args, **kwargs):
    labels = np.unique(y)
    if len(labels) > 2:

        raise ValueError("Can only plot binary classification, got labels %s." % (labels,))
    
    for label in labels:
        ax.plot(x[y==label, 0], x[y==label, 1], '.', color=LABEL_COLORS[label], **kwargs)


def plot_classifier(ax, x, y, model, boundary, res=500, weights=None,bubble_size=CIRCLE_SIZE, invert_incorrect_colors=False, **kwargs):
    # plot samples (different symbols?)
    y_hat = model.predict(x) if hasattr(model, 'predict') else model

    plot_dataset(ax, x, y_hat, **kwargs)
    # mark incorrect symbols, calc accuracy
    false_neg= np.where((y_hat == 1) & (y == -1))[0]
    false_pos = np.where((y_hat == -1) & (y == 1))[0]
    # draw a circle around the false positives & negatives
    fp_color = LABEL_COLORS[1]
    fn_color = LABEL_COLORS[-1]
    if invert_incorrect_colors:
        fp_color, fn_color = fn_color, fp_color
    ax.scatter(x[false_pos, 0], x[false_pos, 1], s=bubble_size, facecolors='none', edgecolors=fp_color)
    ax.scatter(x[false_neg, 0], x[false_neg, 1], s=bubble_size, facecolors='none', edgecolors=fn_color)

    incorrect = np.concatenate([false_pos, false_neg])

    accuracy = 1 - len(incorrect) / len(y)
    x_lim, y_lim = ax.get_xlim(), ax.get_ylim()

    # show decision boundary
    #xx, yy = np.meshgrid(np.linspace(*x_lim, res), np.linspace(*y_lim, res))
    #x, y = xx.flatten(), yy.flatten()
    #X = np.vstack((x, y)).T
    #p = model.predict(X)
    #p = p.reshape(res, res)
    if hasattr(model, 'plot'):
        model.plot(ax)

    # display = DecisionBoundaryDisplay.from_estimator(model._model, X, ax=ax)
    # display.plot(ax=ax, xticks=[], yticks=[])
    #ax.contour(xx, yy, p, alpha=0.5, colors='w', levels=[0.5])
    # ax.imshow(p, extent=(x_lim[0], x_lim[1], y_lim[0], y_lim[1]), origin='lower', cmap='coolwarm', alpha=0.3)

    # show true boundary
    # ax.plot(boundary[1:-1, 0], boundary[1:-1, 1], 'r-')
    ax.set_aspect('equal')
    ax.yaxis.set_visible(False)
    ax.xaxis.set_visible(False)
    return accuracy

