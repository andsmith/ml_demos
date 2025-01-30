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
1. Construct a ***similarity graph***, the possibly weighted, undirected graph $W=(V, E)$ where each vertex represents a sample in $X$ and edges between vertices are defined by the similiarity function using something like one of these methods:
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

2. Construct the ***Laplacian matrix*** from the edge weight matrix.  The laplacian $L$ is the same as the weight matrix $W$ in the off-diagonal elements, but with the diagonal set to the negative sum of each row of W (or column, since W is symmetric):

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

  ### The Canvas, Toolbar, and Similarity graph (weight matrix):
  
![datasets](/spectral_clustering/assets/cluster_UI_and_toolbar.png)

The tools at the bottom let the user choose:
*  A **Cluster Type** to draw a cluster on the canvas.  The center control point can be used to move clusters, the larger axis control point to resize/rotate the cluster, and the smaller to change its aspect ratio.  The "Num Pts" slider sets all clusters to contain that number of points.  (And new ones will be created with that number.)  Dragging a cluster's center out of bounds will delete that cluster.  These four clusters can be added to the canvas:
    * Gaussian - A 2-d multivariate normal with an ellipse at 1-sd.
    * Ellipse - A uniform distribution within the boundary.
    * Annulus - Ring of points.  Experiment with clusters inside clusters (see below).
    * Sierpinski - The fractal.  Create a set of points for which many possible numbers of clusters (values of $k$) are equally valid.  Observe the results in the spectrum.

* One of the **Sim Graph** types, described in the previous section.  It's parameter area is above the "Run" and "Clear" button.  The image shows the "Full" similarity graph is selected and its parameter $\sigma$ is set to 43.859 (units are raw pixel distances).

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

### Hotkeys

```
      Hotkeys:
              'g' - toggle graph view
              ' ' - toggle cluster controls
              'r' - recalculate clustering
              'c' - clear
              'q' - quit
              'h' - print hotkeys
```
Cluster controls are the axis/boundary lines and three control points of each cluster.  See below for an example of "Graph view" on with cluster controls off.


## Things to try:

Classic cases that confuse K-means are non-convex clusters, for example these concentric rings:

  ![datasets](/spectral_clustering/assets/contest_points.png)

K-means cannot recover the natural clusters since it relies on convex partitions:

  ![datasets](/spectral_clustering/assets/kmeans_fail.png)

whereas spectral clustering can because it looks for partitions that have points with mutually high similarity:

  ![datasets](/spectral_clustering/assets/spectral_win.png)


## The Graph View:

The hotkey 'g' toggles showing the edges that exist between vertices given current parameter values for the binary similarity graph methods, N-neighbors and epsilon.  (It is very slow and not useful for the other two).  For the concentric rings example, using the N-neighbors graph with N=8 and allowing non-mutual neighbors, (and with cluster controls off) this looks like:

![datasets](/spectral_clustering/assets/graph_view.png)

The parameters indicated in the toolbar explain the success of the spectral algorithm:  it succeeded in recovering the three concentric clusters because parameter settings created a graph with only three edges between points in different clusters (so the clusters are easily separable) and where every point has several edges to other points in the same cluster (so there are no stragglers/outliers).


# Experiments on the MNIST handwritten digit dataset:

The [MNIST digit classification dataset](https://keras.io/api/datasets/mnist/) consists of 70,000 greyscale images handwritten digits 0 - 9:

![datasets](/spectral_clustering/assets/MNIST/mnist_data.png)

Treating each as a 784-element vector and its digit as the label, how well do the clusters in this dataset conform to class labels?

### General experiments:
 To explore this, the MNIST scripts contain three types of experiments:
 * **Full clustering**:  If we put all the data together and look for $K=10$ clusters, will the digits be separated by cluster?
 * **Pairwise clustering**:  If we combine the samples from two digits and cluster the result with $K=2$, do the most prominent clusters also separate the digit classes?
 * **Single digit clustering**:  Can we discover subtypes of a single digit, such as crossed and uncrossed sevens?

The general process for comparing clusters to class labels is:
  1. Reduce dimensionality to 30 using PCA on the full dataset.
  2. Cluster for $K=2$ or $10$ clusters, assign cluster--class mapping that results in highest accuracy.
  3. Repeat step 3 to get the best clustering (10 times seems stable).

For pairwise clustering experiments, average accuracies are computed over all 45 pairs.  For full clustering experiments, averages are over 100 random samples of the full dataset.
  

## Clustering digits with K-Means
As a basline, the experiments are run with K-means.  Do the clusters that K-means find look like they separate the digit image classes?

Run `> python mnist_kmeans.py` which will run the pairwise and full (10-digit) clustering experiments and generate the following figures:

#### K-Means pairwise results

When given data from two digits, K-Means finds two clusters that correspond to those two digits with varying degrees of accuracy:

![datasets](/spectral_clustering/assets/MNIST/KM_pariwise_accuracy.png)

The pairs (1, 0) (6, 7), and (6,9) are particularly well separated, whereas the pairs (5, 8), (4, 9), and (7, 9) are more mixed.

The next graph shows a 2-d embedding (down from the 30 PCs) of each of the 45 pairwise cluster experiments, showing the correct and incorrect classifications.  The horizontal axis separating the cluster's centers and the vertical axis maximizing point spread:

![datasets](/spectral_clustering/assets/MNIST/KM_pariwise_clusters.png)

It's more useful zoomed in, e.g. to (6, 8):

![datasets](/spectral_clustering/assets/MNIST/KM_pariwise_clusters_zoomed.png)

The final pairwise experiment plot shows the digit images in their embedded positions, with the correct and incorrect digits in side-by-side plots.  The 3 best and worst clusterings (label correspondences) are plotted:


![datasets](/spectral_clustering/assets/MNIST/KM_pairwise_best_worst.png)

With the more accurate pairs, the errors are on the boundary between clusters.  With the least accurate, they are all mixed together.    

#### K-means full 10-digit results:

Putting all the data together, looking for 10 clusters in it, and assigning them class labels yields much lower accuracy than attempting to distinguish only two digits at a time:


![datasets](/spectral_clustering/assets/MNIST/KM_full_accuracy.png)

The upper plot shows, in cell $(i,j)$ the proportion of digits with label $j$ that ended up in cluster $i$.  The diagonal elements are therefore the fraction of correctly identified digits in each class, and the off-diagonals show the frequent misidentifications.  As with the pairwise experiments, 1 is easiest to distingiush from the other digits and 5 is much harder.

The lower plot shows the distribution of (all digit) accuracies over the 100 randomly sampled trials. 

## Spectral clustering results

#### Tuning parameters - the Similarity Graph:

First we need to make a similarity graph from handwritten digits in PCA(30) space.  Ideally, there are many edges between all the vertices corresponding to images of the same digit, and few between vertices belonging to different classes.  We can use two metrics to measure this:
  * **Number of connected components**:  if there are $C$ connected components in the similarity graph, spectral clustering is looking for $K$ clusters, and $C$ < $K$, then the $K$ clusters will consist of an arbitrary partitioning of the $C$ components, since the laplacian matrix will have $C$ eigenvalues corresponding to $\lambda=0$, and some have been dropped before the final clustering.  
  * **Normalized cut**:  If, at the other extreme, the graph is too connected, there will be many edges between different digits.  This can be measured with the "[normalized cut](https://en.wikipedia.org/wiki/Segmentation-based_object_categorization#Normalized_cuts)" of the similarity graph.   Since we know the digit labels, we can use them to partition the data into separate classes and count how many edges of a given graph cross the partition.  In general:
    * if most of the edges (of the binary graphs--the continuous ones sum edge weights) stay within a partition, the normalized cut will be low, or
    * if a large total mass of edges crosses between partitions, the norm cut will be high.

For a graph $G$, with verticies partitioned into sets $A$ and $B$, the cut of the partitioning is 

$$
\text{NormCut}(\{A,B\}) = \frac{\text{Out}(A)}{\text{Deg}(A)} +\frac{\text{Out}(B)}{\text{Deg}(B)} 
$$
* $\text{Out}(A) = \sum_{i\in A, j\in B} w_{ij}$ (the sum of edge weight from $A$ to $B$)
* $\text{Deg}(A) = \sum_{i\in A, j\in G} w_{ij}$ (the sum of edge weight from $A$ to any vertex).


Run `python mnist_simgraphs.py` to test various values of the different similarity graphs' parameters and generate the following plots:

![datasets](/spectral_clustering/assets/MNIST/simgraph_tuning_pairwise.png)
The left two plots show results using the similarity graphs that have a $K$ or $\alpha$ parameter, values between 1 and 50 being tested.  The two similarity graph types with parameters that scale with the data space are plotted on the right.

The upper two figures show how the number of connected components (y-axis, log scale) shrinks as higher value of each paramter adds edges to the graph.  The lower two figures show the normalized-cut metric. 
### Tuning parameters - Spectral clustering

[ADD EXPERIMENT DESCRIPTION HERE]

Run `> python mnist_tuning.py` to compare the different resulst on the pairwise and full experiments.  This generate the following figure:

[ADD FIGURE & DISCUSSION]


#### Spectral pairwise results:

To compare the results of spectral clustering the digits pairwise using the diferent similarity graph types (and the KMeans results), run `> python mnist_pairwise.py`.  This will generate this figure:

#### Spectral full results:

To compare the four similarity graphs with the KMeans resulsts, run `> python mnist_pairwise.py` to generate this figure:

#### Spectral single digit results:

To find digit sub-classes with the spectral clustering algorithms, run `> python mnist_single.py`.