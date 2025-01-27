"""
Spectral clustering on all 10 digits at once:
  * Do clusters correspond to digits?

To investigate, for each spectral algorithm variant:
  1. cluster the data using K=10
  2. Assign each cluster to a digit using the Hungarian algorithm.
  3. Compute the confusion matrix and a 2-d embedding to show the clusters.

Plot results in 3 figures:
  * Confusion matrix with (mean) accuracy as a title, each algorithm (and K-means) in a subplot.
  * Box-plots of mean accuracies over N trials (random subset of data)
  * The cluster embeddings, each in separate subplots.

"""
