"""
Clustering two digits:  Do the clusters correspond to class labels?

Compare:
  * Spectral clustering
  * K-means
  * Fisher LDA (uses labels, is upper bound for K-means)

To investigate, for each clustering algorithm variant (and k-means):

    1. All 45 pairs of digits are compared:
       a. cluster the data using K=2, 
       b. assign cluster/digit labels that give higher accuracy,
       c. Report accuracy mean/sd over the 45 pairs.

    2. Plot results:
       a. A 10x10 image representing accuracy of each pair & colorbar.
       b. An image divided into a 9x8 grid, each showing a 2-d embedding of the cluster pair.
       c. Cluster embeddings (with digit images) of the N most and least accurate pairs.
"""