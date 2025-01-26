import numpy as np
from sklearn.decomposition import PCA
import cv2
import logging
import os
import pickle

class MNISTData(object):
    """
    Load raw data, reduce dimensionality, randomly sample to create subset for all testing.
    (use same subset)
    """
    CACHE = "MNIST_data_local.pkl"

    def __init__(self, dim=30, n_train=1000, n_test=2000, rnd_seed=42):
        """
        :param dim: number of PCA components to keep
        :param n_train: number of training samples to use PER digit
        :param n_test: number of test samples to use per digit
        """
        self._rnd = np.random.RandomState(rnd_seed)
        self._n_train = n_train
        self._n_test = n_test
        (x_train, y_train), (x_test, y_test) = self._load()
        x_orig = np.vstack((x_train, x_test))
        # reshape and normalize
        x_train = x_train.reshape((x_train.shape[0], -1))
        x_test = x_test.reshape((x_test.shape[0], -1))

        # reduce dimensionality
        logging.info("Computing PCA with {} components, and data {}".format(dim, x_train.shape))
        pca = PCA(n_components=dim)
        if dim == 0:
            x_test_pca = x_test
            x_train_pca = x_train
        else:
            x_train_pca = pca.fit_transform(x_train.astype(np.float32)/255.)
            x_test_pca = pca.transform(x_test.astype(np.float32)/255.)

        logging.info("\tPCA complete.")

        x_raw = np.vstack((x_train_pca, x_test_pca))
        y_raw = np.concatenate((y_train, y_test))

        def sample_digit(digit):
            idx = np.where(y_raw == digit)[0]
            subset = self._rnd.choice(len(idx), n_train+n_test, replace=False)
            images = x_orig[idx[subset]]
            data = x_raw[idx[subset]]
            return data, images

        self._train, self._test, self._test_images, self._train_images = {}, {}, {},{}
        for d in range(10):
            x, img = sample_digit(d)
            self._train[d] = x[:n_train]
            self._test[d] = x[n_train:]
            self._train_images[d] = img[:n_train]
            self._test_images[d] = img[n_train:]

    def _load(self):
        if os.path.exists(self.CACHE):
            logging.info("Loading MNIST data from cache.")
            with open(self.CACHE, "rb") as f:
                return pickle.load(f)
        else:
            # import here because it's slow
            from keras.api.datasets import mnist
            logging.info("Downloading MNIST data.")
            (x_train, y_train), (x_test, y_test) = mnist.load_data()
            with open(self.CACHE, "wb") as f:
                pickle.dump(((x_train, y_train), (x_test, y_test)), f)
            return (x_train, y_train), (x_test, y_test)
        logging.info("\tGot %i training and %i test samples." % (len(y_train), len(y_test)))


    def get_images(self, d):
        return self._train_images[d], self._test_images[d]

    def get_digit(self, d):
        return self._train[d], self._test[d]
    
    def get_n(self):
        return self._n_train, self._n_test

    

def test_data():
    """
    Make a collage of data.
    """
    n_train_rows = 9
    n_train_cols = 5
    n_test_rows = 9
    n_test_cols = 4
    import ipdb; ipdb.set_trace()
    data = MNISTData(dim=0,  # No PCA so we keep the image
                     n_test=n_test_rows*n_test_cols,
                     n_train=n_train_rows*n_train_cols)
    train_imgs, test_imgs = {}, {}
    for d in range(10):
        train, test = data.get_digit(d)
        train_images = [sample.reshape(28, 28) for sample in train]
        test_images = [sample.reshape(28, 28) for sample in test]
        train_imgs[d] = np.vstack([np.hstack(train_images[i*n_train_cols:(i+1)*n_train_cols])
                                  for i in range(n_train_rows)])
        test_imgs[d] = np.vstack([np.hstack(test_images[i*n_test_cols:(i+1)*n_test_cols]) for i in range(n_test_rows)])

    output = np.zeros((28*(n_train_rows+n_test_rows)*3,  4*28*(n_train_cols+n_test_cols)), dtype=np.uint8)
    x, y = 0, 0
    for d in range(10):
        output[y:y+train_imgs[d].shape[0], x:x+train_imgs[d].shape[1]] = train_imgs[d]
        x += train_imgs[d].shape[1]
        output[y:y+test_imgs[d].shape[0], x:x+test_imgs[d].shape[1]] = test_imgs[d]
        x += test_imgs[d].shape[1]
        if x >= output.shape[1]:
            x = 0
            y += train_imgs[d].shape[0]

    title = "MNIST - %i train, %i test per digit (esc to close)" % (n_train_rows*n_train_cols, n_test_rows*n_test_cols)
    cv2.imshow(title, output)
    cv2.waitKey(0)


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)
    test_data()
    logging.info("Done.")
 