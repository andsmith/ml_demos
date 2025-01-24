"""
Cluster MNIST digits using spectral clustering.




Question:  How does the number of clusters affect the clustering results?  What are the "natural"
  clusters in the data when K=2, K=3, K=10, K=20, K=30, ...  Do they conform to the digit classes?
  Handwriting style?

General algorithm details.
    1. Dimensionality reduction using PCA, with n=10, 20, 30, 50, 100

"""