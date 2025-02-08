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
 To explore this, the MNIST scripts contain two types of experiments:
 * **Comparing cluster labels to class labels**:  If we put all the digits together and look for 10 clusters in the data, will it naturally find the 10 digit types?   What if we just put two digits, '0', and '1' together?  What about the other pairs? 
   * To investigate the 10-digit case (the "full" expriments), we cluster samples from all 10 digits looking for $K=10$ clusters, then measure the similarity between the found clusters and class labels.  The "accuracy" reported is of the best bijective mapping between class labels (digit labels) and cluster IDs. 
   * To investigate the pairwise case, a similar experiment is done for each of the 45 pairs of digits:  combining their two sets of samples, clustering looking for $K=2$ clusters, and reporting the "accuracy" as the fraction of correct labels, given whichever mapping (Cluster-0 = Digit-A, and Cluster-1 = Digit-B, or the other way around) yields a higher number.
 * **Single digit clustering**:  Can we discover subtypes of a single digit, such as crossed and uncrossed sevens?  To investigate this, we cluster the samples from each digit separately and report a few examples from each cluster, from each digit.

The general process for comparing clusters to class labels is:
  1. Reduce dimensionality to 30 using PCA on the full dataset.
  2. Cluster for $K=2$ or $10$ clusters, assign cluster--class mapping that results in highest accuracy.

For pairwise clustering experiments, average accuracies are computed over all 45 pairs (using 1000 images of each digit).  For full clustering experiments, averages are over 20 random samples (using 500 images of each digit) of the full dataset.
  

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


#### Connectivity of the Similarity Graph:

This experiment is establish good paramter values for constructing the different kinds of similarity graphs.

 Ideally, there are many edges between all the vertices corresponding to images of the same digit, and few between vertices belonging to different classes.  


To measure this, we use as a metric, the **number of connected components**:  if there are $C$ connected components in the similarity graph, spectral clustering is looking for $K$ clusters, and $C$ < $K$, then the $K$ clusters will consist of an arbitrary partitioning of the $C$ components into the $K$ clusters (i.e. the clustering will always give vertices in the same connected component the same cluster label), since the laplacian matrix will have $C$ eigenvalues corresponding to $\lambda=0$, and some have been dropped before the final clustering.  

Intuitively, this means a fully connective graph should be necessary for a good clustering.

Run `python mnist_simgraphs.py` to test various values of the different similarity graphs' parameters in the 10-digit and pairwise experiments:

![datasets](/spectral_clustering/assets/MNIST/simgraph_tuning_pairwise.png)

(Plot from the 'pairwise' experiment, results are average number of connected components in graphs made from samples over all 45 unique digit pairs.)

The left plot shows results using the similarity graph types that have a $K$ or $\alpha$ parameter, values between 1 and 50 being tested.  The similarity graph types with parameters that scale with the data space are plotted on the right.

Observations:
* All graphs become a single component as their parameter increases (as expected).  
* Graphs that become a single connected component quickly, `soft_neighbors_additive` with $\alpha=1$   and `n-neighbors` with $n=2$, are the more inclusive version of those two graph types.
* Their more exclusive counterparts, `soft_neighbors-multiplicative` and `n-neighbors` mutual both require neighboring vertices to be on each other's nearest lists to form an edge between them, and accordingly become connected more slowly.
* The `full` similarity graph being initially more than one connected component is due to round-off error. 
* The `epsilon` similarity graph requires a threshold almost as large as the largest distance between points to connect the vertices into a single component.  

Conclusions:

The digits should cluster better if the graph is fully connected, for reasons discussed above, so the paramters $K$, $\alpha$, $\epsilon$, and $\sigma$ can be chosen appropriately by consulting the plots above. 

## Spectral clustering results

For these experiments, samples from 2 or 10 digits are concatenated into a single dataset, the algorithm looks for 2 or 10 clusters in the combined data, cluster IDs are mapped to the original labels so as to give the highest accuracy (most resembling class labels), which is plotted over a range of values of the similarity graph parameter.

### Spectral pairwise results - cluster/class relationship:
Run `> python mnist_pairwise.py` to run the experiment and plot the clustering "accuracy":

![datasets](/spectral_clustering/assets/MNIST/spectral_pairwise_accuracy.png)

#### Discussion

The average using K-Means as the clustering algorithm is plotted for comparison.  Interestingly, for some parameter values, the spectral algorithms are better able to separate the digits just by clustering the data better (i.e. without looking at the digit labels).

Cluster IDs begin to resemble class labels as soon as the similarity graph becomes connected, and as further edges are added (as the parameter is increased) the performance is reduced as spurious edges (i.e. between clusters/classes) are added.  As the graph becomes too complete, cluster lables look less like digit labels.

Interestingtly even after becoming complete graphs, the clusters found by `full` and `epsilon` similarity graph never resembled the digit classes.

### Spectral pairwise results - Error gallery:

Which digits were most often classified incorrectly?  Which samples did the clustering algorithm decide belonged in the wrong class?  To see examples of the erroneously classified digits for the different similarity graph types, run the script `> python mnist_pairwise_show_errors.py`

[NOTE: After running `mnist_pairwise.py`, results files should be generated in the `ml_demos/spectral_clustering/results` subdir.  This script will only show results if those files have been generated.]

This shows the accuracy curves from the pairwise experiment, but mousing over a plot allows the user to select a curve (closest to the mouse) and a parameter value (x), indicated by the title and the dotted vertical line:

![datasets](/spectral_clustering/assets/MNIST/spectral_pairwise_error_gui.png)


This will load the results from the similarity graph of the selected type built with the selected parameter value and plot a grid showing which samples (or a subset if there are many): 


![datasets](/spectral_clustering/assets/MNIST/spectral_pairwise_errors.png)

In each (i,j) cell is an image showing the samples that were erroneously classified as the other digit in the pair, separated by a line.  The cells with many samples had poor classifications, and the many samples they contain look like valid digits.  The (i,j) pairs with just a few samples correspond to a good cluster/class correspondence, and the errors look more


### Spectral full results:

To see how class labels compare to natural clusters in the data using all 10 digits at once, run `> python mnist_pairwise.py` to generate a figure simlar to the pairwise plot:


![datasets](/spectral_clustering/assets/MNIST/spectral_full_accuracy.png)

Cluster/class correspondence "accuracy" plotted (with bars at +/- 1 standard deviation) for the different graph types over different parameter values. 

The curves have a similar shape but, interestingly, the performance gain Spectral Clustering seemed to have over K-Means in the pairwise experiment is smaller when trying to separate images of all 10 digits at once. 

### Spectral single digit results:

The single-digit clustering GUI can be started by running `> python mnist_single.py`.

Choose the clustering algorithm, similarity graph (for the `spectral` algorithm option), its parameter value the number of clusters, and the dataset size. 

The figure below shows the selected algorithm, `spectral` clustering, will use a `soft-nearest-additive` similarity graph with parameter $\alpha=17$ and will find $K=3$ clusters in 1000 samples of each digit.

Clicking the `Run` button will cluster each digit and plot sample images from each of the $K$ clusters in each digit's dataset, with the spectrum of the laplacian matrix.  The number of images per cluster can be controled by the slider below, shown set to 25 

![datasets](/spectral_clustering/assets/MNIST/spectral_single_explorer.png)

Discussion:
* Many digits seem to cluster by overall angle.
* The three clusters of 2's include one cluster with the digit always drawn with a loop, and two clusters rarely so.
* Rarer subtypes (crossed 7's, triangular 4's) might appear as clusters with a higher $K$.