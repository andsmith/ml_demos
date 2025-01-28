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

    def __init__(self):
        (x_train, y_train), (x_test, y_test) = load_cached(self._read_mnist,
                                                           "MNIST_raw_data.pkl")

        x = np.vstack((x_train, x_test))
        y = np.concatenate((y_train, y_test))
        self._digits = {d: x[y == d] for d in range(10)}
        self._digits_flat = {d: x[y == d].reshape((len(x[y == d]), -1)) for d in range(10)}
        logging.info("Loaded MNIST data, %i samples." % (len(y), ))

    def get_digit_sample_inds(self, d, n_sample=0):
        """
        Get indices of n_sample samples of digit d.
        :param d: digit
        :param n_sample: number of samples to get. If 0, get all.
        :return: indices of samples
        """
        n = len(self._digits[d])
        if n < n_sample:
            raise ValueError("Too many samples requested.")
        elif n_sample == 0:
            return np.arange(n)
        return np.random.choice(n, n_sample, replace=False)

    def _read_mnist(self):
        # Import here because it's slow and unnecessary if
        # already cached.
        from keras.api.datasets import mnist
        return mnist.load_data()

    def get_digit(self, d, n_sample=0):
        return self._digits_flat[d]

    def get_images(self, d):
        return self._digits[d]


class MNISTDataPCA(MNISTData):
    def __init__(self, dim=30, **kwargs):
        """
        :param dim: number of PCA components to keep
        :param kwargs: passed to sklearn.decomposition.PCA
        """
        self._d = dim
        super(MNISTDataPCA, self).__init__()
        self._reduce_dim()

    def _reduce_dim(self):
        logging.info("Computing PCA with %s components." % (self._d, ))
        self._pca = PCA(n_components=self._d)
        self._pca.fit(np.vstack([self.get_digit(d) for d in range(10)]))

        logging.info("\tPCA complete.")
        self._digits_flat = {d: self._pca.transform(self.get_digit(d)) for d in range(10)}


def test_data(plot=False):
    MNISTData()
    pca_data = MNISTDataPCA(dim=2)
    if not plot:
        return

    import matplotlib.pyplot as plt
    for d in range(10):
        plt.scatter(pca_data.get_digit(d)[:, 0], pca_data.get_digit(d)[:, 1], label=str(d), s=1)
    plt.legend()
    plt.title("MNIST data, 2 principal components")
    plt.show()


def test_data_img():
    """
    Make a collage of data.
    """
    n_rows = 4
    n_cols = 4
    data = MNISTDataPCA(dim=30)

    imgs = {}
    for d in range(10):
        images = data.get_images(d)
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

    title = "MNIST data"
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.imshow(1-output/255., cmap='gray')
    ax.set_title(title)
    # axes off
    ax.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    test_data(True)
    test_data_img()
    logging.info("Done.")
