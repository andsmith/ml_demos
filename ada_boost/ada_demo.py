"""
Show each iteration of adaboost.
"""
import numpy as np
import matplotlib.pyplot as plt
from perceptron import DecisionStump
from spiral import make_bump, make_spiral_data, make_minimal_data
import logging
from plotting import plot_dataset, plot_classifier, POINT_MARKER_SIZE


def weights_to_sizes(weights):
    return 100 * weights / np.max(weights)

def _draw_lr(ax, lr_model, *args, **kwargs):
    coeffs = lr_model._weights
    intercept = lr_model._bias
    if coeffs[1] == 0:
        p0 = np.array([-intercept / coeffs[0], 0])
        p1 = np.array([(-intercept - coeffs[1]) / coeffs[0], 1])
    else:
        p0 = np.array([0, -intercept / coeffs[1]])
        p1 = np.array([1, (-intercept - coeffs[0]) / coeffs[1]])

    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], *args, **kwargs)


class AdaDemo(object):
    """
    At each iteration, add these plots to the current figure row:
        * In the left column shows the classifier so far, the data, and the misclassified points
        * In the right column shows the weighted data and the new classifier added to the ensemble.    


    """

    def __init__(self, n_side_pts=25, n_rows=3):
        self._points, labels = make_minimal_data()
        # make labels -1, 1
        u_labs = np.unique(labels)
        if len(u_labs) != 2:
            raise ValueError("Need exactly two classes")
        
        self._labels = np.zeros(labels.shape)
        self._labels[labels == u_labs[0]] = -1
        self._labels[labels == u_labs[1]] = 1


        #self._points, self._labels = make_bump(n_side_pts, h=.25,w=.2,x_left=.0, noise_frac=.003)
        #self._points, self._labels,_ = make_spiral_data(n_side_pts, turns=2.0, ecc=1.0, margin=0.04, random=False)    
        self._n_pts = self._points.shape[0]
        self._n_rows = n_rows
        plt.ion()
        self._fig, self._axs = plt.subplots(n_rows, 3)
        self._axs = np.atleast_2d(self._axs)
        self._next_row = 0

        self._iter = 0
        self._weights = np.ones(self._n_pts) / self._n_pts
        self._error_rates = []  # error rate of hypotheses i under distribution weights[i]
        self._models = []
        self._alphas = []
        self._loss = []  # loss of ensemble at each iteration
        # self._pre_test()
        self._run()


    def eval_ensenble(self, points=None, sign=True):
        points = points if points is not None else self._points
        preds = np.zeros(points.shape[0])
        for model, alpha in zip(self._models, self._alphas):
            single_predictions = model.predict(points)
            preds += alpha * single_predictions

            
        return np.sign(preds) if sign else preds
    
    def _plot_ensemble(self,row, iter):
        """
        Plot the ensemble of models so far, the data, and the misclassified points.
        """
        ax =self._axs[row, 2]
        y_hat = self.eval_ensenble()
        plot_classifier(ax, self._points, self._labels, model=y_hat,boundary=None, markersize=POINT_MARKER_SIZE)
        self._plot_boundary(ax)
        accuracy = np.mean(y_hat == self._labels)
        ax.set_title('Ensemble accuracy:  %.3f'%( accuracy,))

        

        
    def _plot_weights(self, weights, row, iter):
        # draw weights as circles around each sample
        ax =self._axs[row, 0]
        self._plot_dataset(ax)
        sizes = weights_to_sizes(weights)
        ax.scatter(self._points[:, 0], self._points[:, 1], s=sizes, c='black', alpha=.5)
        ax.set_title('Weights at iteration {}'.format(iter))

    def _plot_weak(self, new_model, w_loss, row, iter):
        ax =self._axs[row, 1]
        #self._plot_dataset(ax)
        plot_classifier(ax, self._points, self._labels, model=new_model, boundary=None, markersize=POINT_MARKER_SIZE)
        new_model.plot(ax)
        ax.set_title('Weak wtd.loss %.3f'%(w_loss,))
        
    
    def _plot_boundary(self, ax):
        """
        Plot the decision boundary of the current ensemble.
        """
        n_pts = 150
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x = np.linspace(x_min, x_max, n_pts)
        y = np.linspace(y_min, y_max, n_pts)
        xx, yy = np.meshgrid(x, y)
        points = np.vstack([xx.ravel(), yy.ravel()]).T
        preds = self.eval_ensenble(points, sign=False)
        preds = preds.reshape(n_pts, n_pts)
        #ax.contourf(xx, yy, preds, cmap=plt.cm.RdBu, alpha=0.8)
        # show predictions as an image, with the color indicating the prediction
        image = ax.imshow(preds, extent=(x_min, x_max, y_min, y_max), origin='lower', cmap='RdBu', alpha=.5)
        
        plt.colorbar(mappable=image, cax=None, ax=ax)
        image.set_zorder(-1)
        ax.set_aspect('equal')

    def _plot_dataset(self, ax):
        """
        u_labs = np.unique(self._labels)
        ax.plot(self._points[self._labels == u_labs[0], 0], self._points[self._labels == u_labs[0], 1],
                 'b.', markersize=POINT_MARKER_SIZE, label='class 0')
        ax.plot(self._points[self._labels == u_labs[1], 0], self._points[self._labels == u_labs[1], 1],
                 'r.', markersize=POINT_MARKER_SIZE, label='class 1')
        """
        plot_dataset(ax, self._points, self._labels, markersize=POINT_MARKER_SIZE)

        ax.set_title('Dataset')

        ax.set_aspect('equal')
        
        
        #ax.get_xaxis().set_visible(False)
        #ax.get_yaxis().set_visible(False)
        
    def _run(self):

        # plot dataset in it's own window before running
        plt.figure()
        self._plot_dataset(plt.gca())
        #plt.show()
        plt.waitforbuttonpress()


        while True:
            # show weights first
            self._plot_weights(self._weights, self._next_row, self._iter)
            

            new_model = DecisionStump()  #
            new_model.fit(self._points, self._labels, sample_weight=self._weights)

        
            new_predictions = new_model.predict(self._points)
            misclassified = new_predictions != self._labels
            weighted_loss = np.sum(self._weights[misclassified])

            # Show new model, its weighted loss.

            self._plot_weak(new_model, weighted_loss, self._next_row, self._iter)
            # calculate alpha
            alpha = .5 * np.log((1 - weighted_loss) / weighted_loss)

            # update weights
            indicators = np.ones(self._n_pts)
            indicators[misclassified] = -1
            exponent = -alpha * indicators
            self._weights *= np.exp(exponent)
            self._weights /= np.sum(self._weights)

            # update ensemble & evaluate
            self._models.append(new_model)
            self._alphas.append(alpha)
            self._error_rates.append(weighted_loss)
            new_loss = self._get_global_loss()
            self._loss.append(new_loss)            
            logging.info('Iteration {}: w_loss = {}, loss = {}, alpha = {}'.format(self._iter, weighted_loss, self._loss[-1] ,alpha))

            # show ensemble so far
            self._plot_ensemble( self._next_row, self._iter)
            plt.show()
            plt.waitforbuttonpress()

            self._iter += 1
            self._next_row += 1
            if self._next_row >= self._n_rows:
                self._fig, self._axs = plt.subplots(self._n_rows, 3)
                
                self._axs = np.atleast_2d(self._axs)
                self._next_row = 0

    def _get_global_loss(self):
        predictions = self.eval_ensenble()
        misclassified = predictions != self._labels
        indicators = np.ones(self._n_pts)
        indicators[misclassified] = -1 
        return np.mean(np.exp(-indicators))

def test_draw():
    points, labels = make_bump(20, .25,.2,.4, noise_frac=.01)
    lr = Perceptron().fit(points, labels)
    
    plt.plot(points[labels == 1, 0], points[labels ==
                                            1, 1], 'r.')
    plt.plot(points[labels != 1, 0], points[labels != 1, 1], 'b.')
    _draw_lr(plt, lr, 'g-')
    plt.show()

            
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    AdaDemo()
    plt.show()
