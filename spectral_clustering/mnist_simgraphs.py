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
from mnist_tuning import MNISTPairwiseTuner, MNISTFullTuner
from scipy.sparse.csgraph import connected_components

# Common params for full and pairwise experiments
DIM = 30  # PCA dimension
N_SAMP = 1000  # per digit
N_BOOT = 20  # bootstraps for error estimation (full only)
N_CPU = 10  # number of CPUs to use for parallel processing
N_VALS = 50  # number of parameter values to test


class SimGraphTunerPairwise(MNISTPairwiseTuner):

    def __init__(self, n_cpu=N_CPU):
        super().__init__(n_param_vals=N_VALS, n_cpu=n_cpu, pca_dim=DIM, n_samp=N_SAMP)
        self._helper_func = _test_g_params

    def _get_prefixes(self):
        return "simgraph tuning", "simgraph_pairwise"  # for caching results

    def plot_results(self):
        """
        self._results is a dict with graph names as keys and values are lists of results from _test_g_params.

        Plot the curves for n_components on the left and the normcut curves on the right
        graphs with K-like parameters above, sigma-like parameters below.

        Plot the mean accuracy over all pairs as a curve over the parameter values, with +/- 1 std dev shaded areas.
        Put  sigma/epsilon on a the lower x-axis and K/alpha on a twin upper X-axis
        """

        # Top x-axis has K-like parameters, bottom x-axis has sigma-like parameters
        axis_assignment = {'n-neighbors_mutual': 'upper',
                           'n-neighbors': 'upper',
                           'soft_neighbors_additive': 'upper',
                           'soft_neighbors_multiplicative': 'upper',
                           'full': 'lower',
                           'epsilon': 'lower'}

        fig, ax = plt.subplots(2, 2, figsize=(10, 8))

        for graph_name, results in self._results.items(): # [['epsilon', self._results['epsilon']]]:#
            param_name = results[0]['param_name']
            param_values = [r['param_value'] for r in results[0]['results']]
            n_trials = len(results)
            n_comps = np.zeros((45, len(param_values)), dtype=np.int32)
            n_comps = np.array([[single_result['n_components']
                               for single_result in pair_results['results']] for pair_results in results])
            norm_cuts = np.array([[single_result['norm_cut']
                                 for single_result in pair_results['results']] for pair_results in results])
            mean_n_comps, st_n_comps = np.mean(n_comps, axis=0), np.std(n_comps, axis=0)
            mean_norm_cuts, st_norm_cuts = np.mean(norm_cuts, axis=0), np.std(norm_cuts, axis=0)
            comp_label = "%s" % graph_name
            if axis_assignment[graph_name] == 'upper':
                print("Plotting %s on axis 0" % param_name)
                comp_axis = ax[0][0]
                ncut_axis = ax[0][1]
            else:
                print("Plotting %s on axis 1" % param_name)
                comp_axis = ax[1][0]
                ncut_axis = ax[1][1]
            
            self._plot_bands(comp_axis, param_values, mean_n_comps, st_n_comps, label=comp_label)
            self._plot_bands(ncut_axis, param_values, mean_norm_cuts, st_norm_cuts, label=comp_label)

        ax[1][0].set_xlabel("sigma/epsilon")
        ax[1][1].set_xlabel("sigma/epsilon")
        ax[0][0].set_xlabel("k/alpha")
        ax[0][1].set_xlabel("k/alpha")

        ax[0][0].set_ylabel("num. connected components")
        ax[1][0].set_ylabel("num. connected components")
        ax[0][1].set_ylabel("normalized cut")
        ax[1][1].set_ylabel("normalized cut")

        ax[0][0].sharex(ax[0][1])
        ax[1][0].sharex(ax[1][1])

        [ax[i][j].grid() for i in range(2) for j in range(2)]
        [ax[i][j].legend() for i in range(2) for j in range(2)]

        plt.suptitle("MNIST data, similarity graph parameter tuning - PAIRWISE\n(PCA-dim=%d, samples/digit=%d, avg over 45 pairs)" % (self._dim, self._n_samples))
        plt.show()


class SimGraphTunerFull(MNISTFullTuner):

    def __init__(self, n_cpu=N_CPU):
        super().__init__(pca_dim=DIM, n_samp=N_SAMP, n_cpu=n_cpu)
        self._helper_func = _test_g_params

    def _get_prefixes(self):
        return "simgraph tuning", "simgraph_full"  # for caching results


def _test_g_params(work):
    """
    :param work: dict with:
        'graph_name': key of GRAPH_TYPES,
        'aux': {'pair': pair},  pair of digits
        'param': (param_name, param_values),
        'inds': (inds0, inds1),  # for locating image
        'data': (x, y), })
    :return: list of results dict.
    """
    graph_name = work['graph_name']
    param_name, param_values = work['param']
    x, y = work['data']
    info = work['aux']
    # inds = work['inds']
    sweep = []
    for param_value in param_values:
        graph = GRAPH_TYPES[graph_name](x, **{param_name: param_value})
        sim = graph.get_matrix()
        graph_cut = _calc_cost(sim, y)
        result = {'norm_cut': graph_cut[0],
                  'n_components': graph_cut[1],
                  'param_name': param_name,
                  'param_value': param_value}
        sweep.append(result)

        print("Graph: %s, %s, param: %s=%s, cut, n_components: %s" % (
            graph_name, info, param_name, param_value, graph_cut))

    return {'results': sweep,
            'graph_name': graph_name,
            'param_name': param_name,
            'aux': info}


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
    # sgt = SimGraphTunerPairwise(n_cpu=1)
    # sgt.run()
    sg = SimGraphTunerPairwise(n_cpu=8)
    sg.run()
    sg.plot_results()
    # SimGraphTunerFull(n_cpu=5).run()
    plt.show()
