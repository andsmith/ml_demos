"""
Common functions for clustering MNIST data to classify digits.

The general process is:
  1. Reduce dimensionality from 28x28 images to 30 using PCA.
  2. Cluster the data using spectral clustering / k-means.
  3. Assign cluster ids to class labels to maximize accuracy.
"""