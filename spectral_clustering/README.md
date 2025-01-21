# Spectral clustering

Sometimes, data points $X$ to be clustered into $K$ clusters are not embedded in a space with an implicit distance function.  Images, documents, etc.  

If a similarity function comparing two samples $S(x_i, x_j)$ can be defined, [spectral clustering](https://en.wikipedia.org/wiki/Spectral_clustering) can determine a clustering (partition) of $X$ such that:

$$
S(x_i,x_j) =
\begin{cases}
 & \text{[high value] if }x_i\text{ and }x_j\text{ are in the same cluster, or}\\
&  \text{[low value]  if }x_i\text{ and }x_j\text{ are in different clusters.}
\end{cases}
$$

The general steps are:
1. Construct a ***similarity graph***, the weighted, undirected graph $W=(V, E)$ where each vertex represents a sample in $X$ and edges (possibly weighted) between vertices are defined by the similiarity function, possibly using one of these ways:
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
  * If there is only one connected component, points with highly weighted edges between them will be close together in the eigenvector-space, so clustering the data 