import numpy as np
import matplotlib.pyplot as plt
from esn import EchoStateNetwork
from multiprocessing import Pool, cpu_count
import sys

import logging


class EquilibriumExplorer(object):

    def __init__(self, n_reps=50,
                 spectral_radii=(.5, .6, .7, .8, .9, .95, .975, 0.99,1.0, 1.1),  # experiment 1
                 res_sizes=(10, 20, 40, 80, 160, 320),
                 n_cpu=0,
                 sparsities_vs_sizes=((0.0, .5, .75, .90, .95,.965, .98),  # experiment 2
                                      (50, 100, 150, 200, 250, 350,500))):
        self._exp_1_sparsity = 0.0
        self._exp_2_spec_rad = 0.99
        self._max_iter = 500

        self._n_cpu = n_cpu if n_cpu > 0 else cpu_count()-1
        logging.info("Running on %i cores" % (self._n_cpu,))
        self._n = n_reps
        self._spec_rads = spectral_radii
        self._res_sizes = res_sizes
        self._sparsities, self._sparse_sizes = sparsities_vs_sizes

        self._results_1 = {'eq_times':  [[[0] for _ in self._spec_rads] for _ in self._res_sizes]}
        self._results_2 = {'eq_times':  [[[0] for _ in self._sparse_sizes] for _ in self._sparsities]}

        self._run_exp_1()
        self._run_exp_2()

    def _run_exp_1(self):
        logging.info('Running (size, spectral_rad) for convergecnce time, with sparsity=%f' % (self._exp_1_sparsity,))
        work = []
        for res_size in self._res_sizes:
            for spec_rad in self._spec_rads:
                work.append((res_size, spec_rad, self._n, self._max_iter, self._exp_1_sparsity))
        results = self._run_exp(work)
        for i, res_size in enumerate(self._res_sizes):
            for j, spec_rad in enumerate(self._spec_rads):
                self._results_1['eq_times'][i][j] = results[i*len(self._spec_rads) + j][2]

    def _run_exp_2(self):
        logging.info("Running (size, sparsity) for convergence time, with spectral_rad=%.3f" % (self._exp_2_spec_rad,))
        work = []

        for sparsity in self._sparsities:
            for res_size in self._sparse_sizes:
                work.append((res_size, self._exp_2_spec_rad, self._n, self._max_iter, sparsity))
        results = self._run_exp(work)
        for i, sparsity in enumerate(self._sparsities):
            for j, res_size in enumerate(self._sparse_sizes):
                self._results_2['eq_times'][i][j] = results[i*len(self._sparse_sizes) + j][2]

    def _run_exp(self, work):
        if self._n_cpu > 1:
            pool = Pool(self._n_cpu)
            results = pool.map(_run_experiment, work)
        else:
            results = [_run_experiment(exp) for exp in work]

        return results

    def plot_results1(self):
        """ 
        Plot mean equilibrium time as a function of reservoir size.
        Show as a curve with error bars.

        In a separate plot show as an image the mean equliibrium time as a function of both variables.
        """
        fig, axes = plt.subplots(1, 3, figsize=(12, 6), sharex=False)
        time_plot, time_img, converged_plot = axes

        eq_times = np.array(self._results_1['eq_times'], dtype=np.float64)

        eq_times[eq_times == self._max_iter] = np.nan
        mean_times = np.nanmean(eq_times, axis=2)
        sd_times = np.nanstd(eq_times, axis=2)

        # Left plot
        n_diverged = 0*eq_times[:, :, 0]
        colormap = plt.cm.hot
        for i, rs in enumerate(self._res_sizes):
            mean_time = mean_times[i]
            std_time = sd_times[i]

            time_plot.plot(self._spec_rads, mean_time, 'o-', label='ESN(%i)' %
                           (rs,), color=colormap(i/len(self._res_sizes)))
            time_plot.fill_between(self._spec_rads, mean_time-std_time, mean_time+std_time,
                                   color=colormap(i/len(self._res_sizes)), alpha=0.3)
            n_diverged[i] = [np.sum(np.isnan(eq_times[i, j])) for j in range(len(self._spec_rads))]

        time_plot.set_xlabel('spectral radius')
        time_plot.set_ylabel('mean iterations to convergence')
        time_plot.set_title('mean iterations vs spectral radius')
        time_plot.legend(loc='upper left')
        ylim = time_plot.get_ylim()
        time_plot.set_ylim([np.max((ylim[0], 0)), np.max((75, ylim[1]))])
        time_plot.set_xlim([np.min(self._spec_rads), np.max((1.0, np.max(self._spec_rads)))])
        time_plot.axvline(x=1.0, color='blue', linestyle='--')

        # center and right plots:

        def _show_img(ax, data, title):
            im = ax.imshow(data.T, interpolation=None, cmap=colormap)
            fig.colorbar(im, ax=ax)
            ax.set_ylabel('spectral radius')
            ax.set_xlabel('reservoir size')
            ax.set_title(title)
            ax.set_yticks(np.arange(len(self._spec_rads)), self._spec_rads)
            ax.set_xticks(np.arange(len(self._res_sizes)), self._res_sizes)
            # add line at 1.0 if it's in view
            line_y = None
            if 1.0 in self._spec_rads:
                line_y = self._spec_rads.index(1.0)
            elif np.max(self._spec_rads) > 1.0:
                ind = np.sum(np.array(self._spec_rads) < 1.0)
                line_y = ind + (1.0 - self._spec_rads[ind])/(self._spec_rads[ind+1] - self._spec_rads[ind])
            if line_y is not None:
                ax.axhline(y=line_y, color='blue', linestyle='--')

        _show_img(time_img, mean_times, 'iterations to convergence\n(excluding diverged trials)')
        _show_img(converged_plot, n_diverged, 'number of diverged trials')

        # converged_plot.show()
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.suptitle('Mean time to fixed point on zero inputs (max_iter=%i, n=%i, sparse=%.2f)' % (
            self._max_iter, self._n, self._exp_1_sparsity))

    def plot_results2(self):
        """
        Plot one curve per sparsity showing the mean time to convergence as a function of reservoir size.
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        colormap = plt.cm.hot
        for i, sparsity in enumerate(self._sparsities):
            eq_times = np.array(self._results_2['eq_times'][i], dtype=np.float64)
            eq_times[eq_times == self._max_iter] = np.nan
            mean_times = np.nanmean(eq_times, axis=1)
            sd_times = np.nanstd(eq_times, axis=1)

            ax.plot(self._sparse_sizes, mean_times, 'o-', label='sparse=%.2f' % (sparsity,),
                    color=colormap(i/len(self._sparsities)))
            ax.fill_between(self._sparse_sizes, mean_times-sd_times, mean_times+sd_times,
                            color=colormap(i/len(self._sparsities)), alpha=0.3)

        ax.set_xlabel('reservoir size')
        ax.set_ylabel('mean iterations to convergence')
        ax.set_title('mean iterations vs reservoir size')
        ax.legend(loc='upper right')
        ax.set_xlim([np.min(self._sparse_sizes), np.max(self._sparse_sizes)])
        ax.set_ylim([0, 75])
        ax.axvline(x=1.0, color='blue', linestyle='--')
        plt.suptitle('Mean time to fixed point on zero inputs (max_iter=%i, n=%i, spec_rad=%.3f)' % (
            self._max_iter, self._n, self._exp_2_spec_rad))


def _run_experiment(args):
    """
    Run N trials of these params
    """
    import logging
    logging.basicConfig(level=logging.INFO)
    res_size, spec_rad, n_reps, max_iter, sparsity = args
    times = []
    for i in range(n_reps):
        esn = EchoStateNetwork(n_input=1, n_reservoir=res_size, n_output=1,
                               spectral_radius=spec_rad, max_eq_iter=max_iter, sparsity=sparsity)
        times.append(esn.n_eq_washout)
    n_bad = np.sum(np.array(times) == max_iter)
    logging.info('res_size=%i, sparsity=%.3f, spec_rad=%f, mean(time)=%.2f, diverged:  %i' %
                 (res_size, sparsity, spec_rad,  np.mean(times), n_bad))
    return (res_size, spec_rad, times)


def find_equilibria():
    if len(sys.argv) > 2:
        sparsity = float(sys.argv[1])
    else:
        sparsity = 0.0
    ee = EquilibriumExplorer()
    ee.plot_results1()
    ee.plot_results2()
    plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    find_equilibria()
