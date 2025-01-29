"""
Explore different similarity graphs on the MNIST data.
Find optimal values for each of their parameters.

Use the "Graph cut" value to predict the best values.

Given true cluster labels we can define the graph cut using the labels as the partition:

    1. For a given graph type and parameter value, construct the similarity graph
    2. Evaluate the normalized cut for the partition labeled by the original labels.
    3. Repeat for all parameter values/graph types.

Run for both pairwise and 10-class tasks.

"""
import logging
import numpy as np
import matplotlib.pyplot as plt
from mnist_data import MNISTDataPCA
from mnist_common import GRAPH_TYPES, GRAPH_PARAM_NAMES
from multiprocessing import Pool, cpu_count
from util import load_cached
from mnist_tuning import MNISTPairwiseTuner
from scipy.sparse.csgraph import connected_components

# Common params for full and pairwise experiments
DIM = 30  # PCA dimension
N_SAMP = 1000  # per digit
N_BOOT = 50  # bootstraps for error estimation
N_CPU = 1  # number of CPUs to use for parallel processing


class SimGraphTunerFull(MNISTPairwiseTuner):

    def __init__(self,n_cpu=N_CPU):
        super().__init__(pca_dim=DIM, n_samp=N_SAMP, n_cpu=n_cpu)
        self._helper_func = _test_g_params

    def _get_prefixes(self):
        return "simgraph tuning", "simgraph_pairwise"  # for caching results


def _test_g_params(work):
    """
    :param work: dict with:
        'graph_name': key of GRAPH_TYPES,
        'aux': {'pair': pair},  pair of digits
        'param': (param_name, param_values),
        'inds': (inds0, inds1),  # for locating image
        'data': (x, y), })
    :return: list of graph_cut values.
    """
    graph_name = work['graph_name']
    param_name, param_values = work['param']
    x, y = work['data']
    pair = work['aux']['pair']
    inds0, inds1 = work['inds']
    results = []
    for param_value in param_values:
        graph = GRAPH_TYPES[graph_name](x, **{param_name: param_value})
        sim = graph.get_matrix()
        graph_cut = _calc_cost(sim, y)
        results.append(graph_cut)
        
        print("Graph:  %s, pair:  %s, param: %s=%s, cut, n_components: %s" % (
            graph_name, pair, param_name, param_value, graph_cut))
    return results

def _calc_cost(sim, y):
    """
    :param sim: similarity matrix
    :param y: true labels
    :return: graph cut cost, i.e. normalized cut:
      cut(graph) = sum_{clusters c} cluster_cost(c), where 
      cluster_cost(c) = (sum_(edges e out of c) w_e) / (sum_(nodes n in c) d_n)
      Where w_e is the edge weight and d_n is the degree of node n.
    """
    # find the number of connected components
    n_components, labels = connected_components(sim)
    # calculate degree matrix
    deg = np.sum(sim, axis=1)
    # calculate cut
    cut = 0
    for c in np.unique(y):
        c_degree = np.sum(deg[y == c])
        # get all nodes in cluster c
        c_mask = (y == c)
        # get all edges from nodes in c
        c_edges = sim[c_mask, :]
        # and and all from nodes in c to nodes not in c
        c_out_edges = c_edges[:, ~c_mask]
        # calculate cluster cost
        cluster_cost = np.sum(c_out_edges) / c_degree if c_degree > 0 else 0  # add self-weight to fix
        cut += cluster_cost

    return cut, n_components
    

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sgtf = SimGraphTunerFull(n_cpu=1)
    sgtf.run()
    plt.show()
