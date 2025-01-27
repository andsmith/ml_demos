"""
What sub-types of individual digits are there in the MNIST dataset?
Can the digits with sub-types be automatically distinguished from digits all drawn the same way?

To investigate, cluster each digit with varying K and, for each digit, plot::
  * The clusters in a 2-d embedding that keeps cluster centers far apart. (2 principle components)
  * Plot lowest 2K eigenvalues next to the embedding.
  * Show each cluster's "prototype", the sample closest to its center of mass in eigenspace.
"""