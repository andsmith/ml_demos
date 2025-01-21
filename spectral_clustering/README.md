# Spectral clustering

Sometimes, a set of objects $X = (x_1,...,x_n)$ to be clustered into $k$ clusters are not embedded in a space with an implicit distance function.  Images, documents, etc.  

If, instead of a natural distance function, a domain-specific *similarity function* $S(x_i, x_j)$ comparing two objects can be defined, [spectral clustering](https://en.wikipedia.org/wiki/Spectral_clustering) can determine a clustering (partition) $P = {P_0, ..., P_k}$ of $X$ such that:

$$
\begin{align*}
S(x_i,x_j) = & \text{[high value] }  \Leftrightarrow \exists c \in [1, k] \mid x_i, x_j \in P_c,  \text{ and} \\
S(x_i,x_j) = & \text{[low value] }  \Leftrightarrow \nexists c \in [1, k] \mid x_i, x_j \in P_c.\\
\end{align*}
$$
That is, given a similarity function, spectral clustering attempts to find the partitioning that results in objects in the same cluster having high similarity and objects in different clusters having low similiarity.  

The general steps are:
1. Construct a ***similarity graph***, the weighted, undirected graph $W=(V, E)$ where each vertex represents a sample in $X$ and edges (possibly weighted) between vertices are defined by the similiarity function, possibly using one of these ways. :
  - **Epsilon threshold**: Let an edge between vertices $i$ and $j$ exist with weight 1 if $S(x_i, x_j) >\epsilon$, for some threshold $\epsilon$.
  - **N Nearest neighbors**: Let an edge between $i$ and $j$ exist with weight  if vertex $j$ is among the $N$ most similar to vertex $i$ or vice-versa, but not necessarily both.  If the condition is required in both direction, this is called "mutual" nearest neighbors.  
  - **Soft Nearest Neighbors**:  Define the parameter $\alpha$ and define the *directed* graph with weights

$$
R[i,j]= e^{-\text{Rank}_i(j)^2/{\alpha^2}}
$$

   Where $\text{Rank}_i(j)$ is the index of vertex $j$ in list of vertices sorted in decreasing similarity to vertex $i$.  I.e. $\text{Rank}_i(j) < \text{Rank}_i(k)$ means $S(x_i,x_j) > S(x_i, x_k)$. Then the weights of the similarity graph are either $R \cdot R^\intercal$ or $R+R^\intercal$ (analogous to mutual and regular N nearest neighbors respectively).

  - **Full** graph: Define the parameter $\sigma$ and let an edge exist between every vertex, and give it a weight that falls off quickly as similarities shrink (controled by $\sigma$):
  
$$
W[i,j] = e^{-{S(x_i, x_j)^2}/{2\sigma^2}}, or
$$

2. Construct the ***Laplacian matrix*** from the edge weight matrix.  The laplacian $L$ is the same as the weight matrix $W$ in the off-diagonal elements, but with the diagonal set to the sum of each row of W (or column, since W is symmetric):

$$
L[i,j] = W - W\cdot\mathbb{1}_{|W|},
$$

Where $\mathbb{1}_{|W|}$ is the column vector of 1's with as many entries as $W$ has rows/cols.  Since the diagonals of $W$ are zero, each row and column of $L$ should now sum to zero.

3. Find the eigenvalues of $L$ sorted in increasing order:

  * The smallest eigenvalue will be zero and its multiplicity $m_0$ will be the number of connected components in the graph $W$.  
  * The corresponding first $m_0$ eigenvectors can be used as $m_0$-dimensional coordinates to cluster the points in euclidean space (e.g. with [K-Means](https://en.wikipedia.org/wiki/K-means_clustering)).
  * If there is only one connected component, as is the case when clusters overlap, points with highly weighted edges between them will be close together in the eigenvector-space, so clustering the data in that representation can still be useful.

  To understand why, read [the tutorial here.](https://people.csail.mit.edu/dsontag/courses/ml14/notes/Luxburg07_tutorial_spectral_clustering.pdf)

  To explore how this works, run the...
  ## Demo

  The clustering demo shows the effects of different similarity graph construction methods on the clustering results.  The user can:
  * Create "clusters", generating random points with various distributions.
  * Select different similarity graph types and their parameters.
  * View the graph, similarity matrix, eigenvalues/vectors, data projected into eigenvector space and the final clustering results.

Run:  `> python cluster_creator.py` to start the demo.

  ### The Canvas, Toolbar, and Similarity graph:
  
![datasets](/spectral_clustering/assets/cluster_UI_and_toolbar.png)

The tools at the bottom let the user choose:
*  A **Cluster Type** to draw a cluster on the canvas.  The central control point can be used to move clusters, the larger peripheral one to resize/rotate it, and the smaller to change the aspect ratio.  The "Num Pts" slider sets all clusters to contain that number of points.  (And new ones will be created with that number.)  Dragging a cluster's center out of bounds will delete that cluster.
    * Gaussian - A 2-d multivariate normal with an ellipse at 1-sd.
    * Ellipse - A uniform distribution within the boundary.
    * Annulus - Ring of points.  Experiment with one cluster inside the other (see below).
    * Sierpinski - The fractal.  Create a set of points for which many possible numbers of clusters are equally valid.  Observe the results in the spectrum.

* One of the **Sim Graph** types, described in the previous section.  It's parameter area is above the "Run" and "Clear" button.  The image shows the "Full" similarity graph is selected and its parameter $\sigma$ is set to 31.613.

* The Clustering **Algorithm** and its parameters.  The algorithms availiable are spectral clustering and K-means.  The spectral clustering algorithm uses these parameters (K-means only uses $K$)
  * $F$ is the number of eigenvectors (i.e. those corresponding to the $F$ smallest eigenvalues) spanning the space the points will be embedded into for clustering.
  * $K$ is the number of clusters to look for in that space (or in the original 2-d space in the case of K-means).

Clicking the `Run` button puts all the points together in a single list for the clustering algorithm to partition.

This populates the four windows to the right of the UI window:

  ### The edge weight histogram, eigenvalues/vectors, and results:
  ![datasets](/spectral_clustering/assets/cluster_results.png)

This shows the output. At top left, the eignevalues with four near zero corresponding to the four clusters in the data.  They are not exactly zero because we are using the "Full" similarity graph, and nonzero edge weights exist between points in different clusters.  At top right, the corresponding eigenvalues.  The red line in both graphs indicates how many eigenvectors are used to re-represent the data, i.e. $F$.  The slider between them can be used to plot more than $F$.

 The bottom-left graph shows the distribution of weight values, which is useful in tuning the parameters.  And the bottom right window shows the final clustering result, a partitioning of the points the user added.

### The Random Projection Window

After the data points are embedded in an F-dimensional space, they are projected down to 3-random (orthogonal) directions in this space and plotted in an interactive window:  
![datasets](/spectral_clustering/assets/cluster_randproj.png)

## Things to try:

Classic cases that confuse K-means are non-convex clusters, for example these concentric rings:

  ![datasets](/spectral_clustering/assets/contest_points.png)

K-means has no hope of recovering the natural clusters since it relies on convex partitions:
  ![datasets](/spectral_clustering/assets/kmeans_fail.png)

whereas spectral clustering can:
  ![datasets](/spectral_clustering/assets/spectral_win.png)


## The Graph View:

The hotkey 'g' toggles showing the edges that exist between vertices given current paramete values for the binary similarity graph methods, N-neighbors and epsilon.  (It is very slow and not useful for the other two):

![datasets](/spectral_clustering/assets/graph_view.png)

The parameters indicated in the toolbar explain the success of the spectral algorithm:  it succeeded in recovering the three concentric clusters because the values for the algorithm parameters (using an N-nearest sim. graph with N=8 and non-mutual edges allowed) created a graph with no edges between vertices in different clusters, and at least one edge from every point to another point in the same cluster.
