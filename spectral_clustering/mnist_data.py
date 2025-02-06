import matplotlib.pyplot as plt
import numpy as np
import cv2
import logging
import os
import pickle
from util import load_cached
import sys
from sklearn.decomposition import PCA


class MNISTData(object):
    """
    Load data, combine test/train sets. 
    """

    def __init__(self, pca_dim=0):
        self._pca_dim = pca_dim
        (x_train, y_train), (x_test, y_test) = load_cached(self._read_mnist,
                                                           "MNIST_raw_data.pkl")
        self.train = {d: np.array([img.reshape(-1) for img in x_train[y_train == d]]) for d in range(10)}
        self.test = {d: np.array([img.reshape(-1) for img in x_test[y_test == d]]) for d in range(10)}
        self.print_digit_stats()

    def print_digit_stats(self):
        """
        print number of training and test images per digit.
        """
        logging.info("MNIST data digit stats:")
        for d in range(10):
            logging.info("\t%i: train %i, test %i" % (d, self.train[d].shape[0], self.test[d].shape[0]))

    def _read_mnist(self):
        # Import here because it's slow and unnecessary if
        # already cached.
        from keras.api.datasets import mnist
        return mnist.load_data()

    def get_sample(self, n_train, n_test=0, digits=None):
        """
        Get random samples.
        """
        return MNISTSample(self, digits,
                           pca_dim=self._pca_dim,
                           n_train=n_train,
                           n_test=n_test)


class PCATransf(object):
    def __init__(self, n_components,whiten=False):
        self._dim = n_components
        self._proj_axes = None
        self.explained_variance_ratio_ = None
        self._whiten = whiten   

    def fit(self, x):
        pca = PCA(n_components=self._dim, whiten=self._whiten)
        pca.fit(x)
        self.explained_variance_ratio_ = pca.explained_variance_ratio_
        self._proj_axes = pca.components_

    def transform(self, x):
        return x.dot(self._proj_axes.T)


class MNISTSample(object):
    """
    Class to hold a test/train split of MNIST data and manage PCA (i.e. for a single trial)
    Also, keep track of sample indices so the image can be retrieved from the full data.
    (Don't store anything large, since we want to pickle this object.)
    """

    def __init__(self, data, digits, pca_dim, n_train, n_test):
        """
        :param data: MNISTData object
        :param digits: iterable of digits to include
        :param pca_dim: PCA dimension (computed on training set of each sample)
        :param n_train: number of training samples per digit
        :param n_test: number of test samples per digit, or -1 for all  
        """
        self.dim, self.pca_dim = data.train[0].shape[1], pca_dim
        self.digits = digits if digits is not None else tuple([i for i in range(10)])

        self.train_inds = {d: np.random.choice(data.train[d].shape[0], n_train, replace=False)
                           for d in self.digits}
        if n_test == -1:
            self.test_inds = {d: np.arange(data.test[d].shape[0], dtype=np.int32) for d in self.digits}
        else:
            self.test_inds = {d: np.random.choice(data.test[d].shape[0], n_test, replace=False)
                              for d in self.digits}

        if self.pca_dim > 0:
            self.pca_transf = PCATransf(n_components=self.pca_dim, whiten=False)  # few percent better without whitening
            train_x = np.vstack([data.train[d][self.train_inds[d]] for d in self.digits])
            logging.info("Computing PCA with %s components on %i points." % (self.pca_dim, train_x.shape[0]))
            self.pca_transf.fit(train_x)
            var_sum = np.sum(self.pca_transf.explained_variance_ratio_[:self.pca_dim])
            sd_range = np.sqrt(self.pca_transf.explained_variance_ratio_[0]), np.sqrt(self.pca_transf.explained_variance_ratio_[-1])
            logging.info("\tPCA complete, sd[0]=%.3f, sd[%i]=%.5f, %.3f %% of variance." %
                         (sd_range[0], self.pca_dim, sd_range[1], var_sum*100))

            self.train = {d: self.pca_transf.transform(data.train[d][self.train_inds[d]]) for d in self.digits}
            self.test = {d: self.pca_transf.transform(data.test[d][self.test_inds[d]]) if self.test_inds[d].size > 0
                         else np.zeros((0, self.pca_dim))
                         for d in self.digits}
        else:
            self.train = {d: data.train[d][self.train_inds[d]] for d in self.digits}
            self.test = {d:  data.test[d][self.test_inds[d]] if self.test_inds[d].size > 0 else np.zeros((0, self.dim))
                         for d in self.digits}
            self.pca_transf = None

    def get_images(self,d,data, inds, which='train'):
        """
        :param d: digit
        :param data: MNISTData object
        :param inds: indices of the images to retrieve (inton self.train or self.test
        :param which: 'train' or 'test'
        :return: list of images
        """
        if which == 'train':
            return data.train[d][self.train_inds[d][inds]].reshape(-1, 28, 28)
        elif which == 'test':
            return data.test[d][self.test_inds[d][inds]].reshape(-1, 28, 28)
        else:
            raise ValueError("Unknown data type: %s (should be 'test' or 'train')" % which)
        
    def get_data(self, which='train'):
        """
        Get the test/training data in single arrays.
        :param which: 'train' or 'test'
        :returns X, y, where X is the data (N x PCA-dim) and y is the digit label (N), where
        N is (n_train or n_test) * (lenght of self.digits).
        """

        def _get_pair(source):
            x = np.vstack([source[d] for d in self.digits])
            y = np.hstack([d*np.ones(source[d].shape[0], dtype=np.int32) for d in self.digits])
            return x, y

        if which == 'train':
            return _get_pair(self.train)
        elif which == 'test':
            return _get_pair(self.test)
        else:
            raise ValueError("Unknown data type: %s (should be 'test' or 'train')" % which)


def test_data(plot=False):
    fig, ax = plt.subplots()
    pca_data = MNISTData(pca_dim=2).get_sample(2000, 0)
    if not plot:
        return
    for d in range(10):
        x = pca_data.train[d]
        ax.scatter(x[:, 0], x[:, 1], label=str(d), s=1)
    plt.legend()
    plt.title("MNIST data, 2 principal components")


def test_data_img(plot=True):
    """
    Make a collage of data.
    """
    n_rows = 4
    n_cols = 4
    data = MNISTData(pca_dim=0).get_sample(100, 0)

    imgs = {}
    n_imgs_per_digit = n_rows*n_cols
    for d in range(10):
        images = data.train[d][:n_imgs_per_digit].reshape(-1, 28, 28)
        imgs[d] = np.zeros((28*n_rows, 28*n_cols), dtype=np.uint8)
        for i in range(n_rows):
            for j in range(n_cols):
                imgs[d][i*28:(i+1)*28, j*28:(j+1)*28] = images[i*n_cols + j]

    output = np.zeros((28*(n_rows)*3,  4*28*(n_cols)), dtype=np.uint8)
    x, y = 0, 0
    for d in range(10):
        output[y:y+imgs[d].shape[0], x:x+imgs[d].shape[1]] = imgs[d]
        x += imgs[d].shape[1]
        if x >= output.shape[1]:
            x = 0
            y += imgs[d].shape[0]
    if plot:

        title = "MNIST data"
        fig, ax = plt.subplots()
        ax.imshow(1-output/255., cmap='gray')
        ax.set_title(title)
        # axes off
        ax.axis('off')
        plt.tight_layout()


def test_all(plot=False):
    # test pca
    MNISTData(30).get_sample(1000, 100)
    MNISTData(50).get_sample(1000, 100)
    MNISTData(100).get_sample(1000, 100)
    # show sample digits
    test_data_img(plot=plot)
    # show data on 2 pca axes
    test_data(plot=plot)
    if plot:
        plt.show()


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    logging.info("Testing MNIST data.")
    test_all(plot=True)
    logging.info("Done.")
