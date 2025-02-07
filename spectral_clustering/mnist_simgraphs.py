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
from mnist_common import GRAPH_TYPES, GRAPH_PARAM_NAMES
from multiprocessing import Pool, cpu_count
from util import load_cached
from mnist_pairwise import MNISTPairwiseTuner
from mnist_full import MNISTFullTuner
from scipy.sparse.csgraph import connected_components

# Common params for full and pairwise experiments
DIM = 30  # PCA dimension
N_BOOT = 16  # bootstraps for error estimation (full only)


class SimGraphResultPlotter(object):
    def _get_param_range(self, graph_name, param_name, data):
        """
        Determine a good set of test values.
        """

        if param_name == 'k':
            # for the nearest neighbor sim graphs
            values = np.arange(1, 101).astype(int)

        elif param_name == 'alpha':
            # for the soft nearest neighbors graphs
            values = np.arange(1, 101)

        elif param_name in ['epsilon', 'sigma']:
            # for epsilon & full graphs (hard/soft thresholding on euclidean distance)
            x = np.vstack([data.train[i] for i in range(10)])
            n_s = 100000
            sample_pairs_a = np.random.choice(x.shape[0], n_s, replace=True)
            sample_pairs_b = np.random.choice(x.shape[0], n_s, replace=True)
            valid = sample_pairs_a != sample_pairs_b
            sample_pairs_a = sample_pairs_a[valid]
            sample_pairs_b = sample_pairs_b[valid]
            distances = np.linalg.norm(x[sample_pairs_a] - x[sample_pairs_b], axis=1)
            # fig, ax = plt.subplots()
            # ax.hist(distances, bins=100)
            # plt.show()
            if param_name == 'epsilon':
                val_range = np.min(distances)/100, np.percentile(distances, (20))
            else:  # sigma
                val_range = np.min(distances)/100, np.percentile(distances, (20))

            values = np.linspace(val_range[0], val_range[1], 50)  # don't go too low

        else:
            raise ValueError("Unknown param name: %s" % param_name)

        logging.info("%s range: %f to %f  (%i values)" % (param_name, values[0], values[-1], values.size))
        return values

    def plot_results(self):
        """
        self._results is a dict with graph names as keys and values are lists of results from _test_g_params.

        Plot the curves for n_components on the left and the normcut curves on the right
        graphs with K-like parameters above, sigma-like parameters below.

        Plot the mean accuracy over all pairs as a curve over the parameter values, with +/- 1 std dev shaded areas.
        Put  sigma/epsilon on a the lower x-axis and K/alpha on a twin upper X-axis
        """
        left = 0
        right = 1
        x_lim={left:250, right:None}

        # Top x-axis has K-like parameters, bottom x-axis has sigma-like parameters
        axis_assignment = {'n-neighbors_mutual': 'left',
                           'n-neighbors': 'left',
                           'soft_neighbors_additive': 'left',
                           'soft_neighbors_multiplicative': 'left',
                           'full': 'right',
                           'epsilon': 'right'}

        fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
        ax = np.array([[ax[0], ax[1]]]) # make it 2D for easier indexing
        for graph_name, results in self._results.items():  # [['epsilon', self._results['epsilon']]]:#
            if results is None:
                logging.info("Skipping graph '%s' - no results found." % graph_name)
                continue
            param_name = results[0]['param_name']
            param_values = [r['param_value'] for r in results[0]['results']]
            n_trials = len(results)
            n_comps = np.array([[single_result['n_comp']
                                for single_result in result_row['results']] for result_row in results])
            norm_cuts = np.array([[single_result['norm_cut']
                                   for single_result in result_row['results']] for result_row in results])
            mean_n_comps, st_n_comps = np.mean(n_comps, axis=0), np.std(n_comps, axis=0)
            mean_norm_cuts, st_norm_cuts = np.mean(norm_cuts, axis=0), np.std(norm_cuts, axis=0)

            comp_label = "%s" % graph_name
            if axis_assignment[graph_name] == 'left':
                print("Plotting %s on axis 0" % param_name)
                comp_axis = ax[0][left]
                #ncut_axis = ax[1][left]
            elif axis_assignment[graph_name] == 'right':
                print("Plotting %s on axis 1" % param_name)
                comp_axis = ax[0][right]
                #ncut_axis = ax[1][right]
            else:
                raise ValueError("Unknown axis assignment for %s" % graph_name)

            self._plot_bands(comp_axis, param_values, mean_n_comps, st_n_comps, label=comp_label)
            #self._plot_bands(ncut_axis, param_values, mean_norm_cuts, st_norm_cuts, label=comp_label)

        for row in range(1):
            ax[row][right].set_xlabel("sigma/epsilon")
            ax[row][left].set_xlabel("k/alpha")
            ax[row][left].set_xlim(1, x_lim[left])

            # set left axis to log-y
        ax[0][left].set_xscale('log')
        ax[0][right].set_xscale('log')

        for col in range(2):
            ax[0][col].set_ylabel("num. connected components")
            #ax[1][col].set_ylabel("normalized cut")
            #ax[0][col].sharex(ax[1][col])

        [ax[i][j].grid() for i in range(1) for j in range(2)]
        [ax[i][j].legend() for i in range(1) for j in range(2)]

        plt.suptitle(self._plot_title)


class SimGraphTunerPairwise(SimGraphResultPlotter, MNISTPairwiseTuner):

    def __init__(self, n_cpu=10):
        super().__init__(n_cpu=n_cpu, pca_dim=DIM, no_compute=False)
        self._helper_func = _test_g_params
        self._plot_title = "MNIST data, similarity graph parameter tuning - PAIRWISE\n(PCA-dim=%d, samples/digit=%d, avg over 45 pairs)" % (
            self._dim, self._n_samples)

    def _get_prefixes(self):
        return "simgraph tuning", "simgraph_pairwise"  # for caching results


class SimGraphTunerFull(SimGraphResultPlotter, MNISTFullTuner):

    def __init__(self, n_cpu=5):
        super().__init__(pca_dim=DIM, n_cpu=n_cpu, n_boot=N_BOOT,n_samp=(500,500))
        self._helper_func = _test_g_params
        self._plot_title = "MNIST data, similarity graph parameter tuning - FULL\n(PCA-dim=%d, samples/digit=%d, avg over %i random samples)" % (
            self._dim, self._n_samples, self._n_boot)

    def _get_prefixes(self):
        return "simgraph tuning", "simgraph_full"  # for caching results


def _test_g_params(work):
    """
    :param work: dict with:{'graph_name': string,
                            'aux': {'pair': (a,b)},
                            'data': MNISTSample,
                            'param': (str: param_name, list:  param_vals),
                            'k': 2, }
                        )
    :return: list of results dict.
    """
    graph_name = work['graph_name']
    param_name, param_values = work['param']
    x, y = work['data'].get_data('train')
    aux = work['aux']
    # inds = work['inds']
    sweep = []
    for param_value in param_values:
        graph = GRAPH_TYPES[graph_name](x, **{param_name: param_value})
        sim = graph.get_matrix()
        cost_results = _calc_cost(sim, y)
        print("Graph: %s, digits: %s, param: %s=%s, cut, n_cut, n_components: %s" % (
            graph_name, aux, param_name, param_value, cost_results))
        cost_results['param_name'] = param_name
        cost_results['param_value'] = param_value
        sweep.append(cost_results)

    return {'results': sweep,
            'graph_name': graph_name,
            'param_name': param_name,
            'aux':aux}


def _calc_cost(sim, y):
    """
    :param sim: similarity matrix
    :param y: true labels
    :return: cut, norm_cut, n_components
        if graph G is partitioned into partitions p_i, 
        then the cut of partition pi is:
           cut(p_i) = sum_(edges e out of p_i) w_e
        the normalized cut is:
           norm_cut(p_i) = (sum_(edges e out of p_i) w_e) / (sum_(nodes n in p_i) d_n)

        And the (norm)cut value for the partitioning on G is the sum of the cut values for each partition.

      Where w_e is the edge weight and d_n is the degree of node n.
    """
    # find the number of connected components
    n_components, labels = connected_components(sim)
    # calculate degree matrix
    deg = np.sum(sim, axis=1)

    # calculate cut
    cut = 0
    norm_cut = 0
    for c in np.unique(y):
        c_degree = np.sum(deg[y == c])
        # get all nodes in cluster c
        c_mask = (y == c)
        # get all edges from nodes in c
        c_edges = sim[c_mask, :]
        # and and all from nodes in c to nodes not in c
        c_out_edges = c_edges[:, ~c_mask]
        # calculate cluster cost
        cut_i = np.sum(c_out_edges)
        norm_cut += cut_i / c_degree if c_degree > 0 else 0  # add self-weight to fix
        cut += cut_i

    return {'cut': cut, 'norm_cut':  norm_cut, 'n_comp': n_components}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    sg = SimGraphTunerPairwise(n_cpu=14)
    sg.run()
    sg.plot_results()
    
    sf = SimGraphTunerFull(n_cpu=4)
    sf.run()
    sf.plot_results()

    plt.show()
