import numpy as np
import matplotlib.pyplot as plt
from esn import EchoStateNetwork
from multiprocessing import Pool, cpu_count


class EquilibriumExplorer(object):

    def __init__(self, n_reps=200,
                 spectral_radii=(.5, .6, .7, .8, .9, .95, .975, 0.99, 1.0, 1.1, 1.3,1.5),
                 res_sizes=(10, 20, 40, 80,160),
                 n_cpu=0):

        self._max_iter = 3000
        self._n_cpu = n_cpu if n_cpu > 0 else cpu_count()-1
        print("Running on %i cores" % (self._n_cpu,))
        self._n = n_reps
        self._spec_rads = spectral_radii
        self._res_sizes = res_sizes
        self._esn_kwargs = {}  # input/output scale will change eq time via the bias, set here?

        self._eq_times = [[[0] for _ in self._spec_rads] for _ in self._res_sizes]

        self._run()

    def _run(self):

        work = []

        for i, res_size in enumerate(self._res_sizes):
            for j, spec_rad in enumerate(self._spec_rads):
                work.append((res_size, spec_rad, self._n, self._max_iter))

        if self._n_cpu > 1:
            pool = Pool(self._n_cpu)
            results = pool.map(_run_experiment, work)
        else:
            results = [_run_experiment(exp) for exp in work]

        for i, res_size in enumerate(self._res_sizes):
            for j, spec_rad in enumerate(self._spec_rads):
                self._eq_times[i][j] = results[i*len(self._spec_rads) + j][2]

    def plot_results(self):
        """ 
        for each spectral radius:
            Plot mean equilibrium time as a function of reservoir size.
            Show as a curve with error bars.

        In a separate plot show as an image the mean equliibrium time as a function of both variables.
        """
        fig, axes = plt.subplots(1, 3, figsize=(12, 6), sharex=False)
        time_plot, time_img, converged_plot = axes

        eq_times = np.array(self._eq_times, dtype=np.float64)

        eq_times[eq_times == self._max_iter] = np.nan
        mean_times = np.nanmean(eq_times, axis=2)
        sd_times = np.nanstd(eq_times, axis=2)

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
        time_plot.legend()

        def _show_img(ax, data, title):
            im = ax.imshow(data.T, interpolation=None, cmap=colormap)
            fig.colorbar(im, ax=ax)
            ax.set_ylabel('spectral radius')
            ax.set_xlabel('reservoir size')
            ax.set_title(title)
            ax.set_yticks(np.arange(len(self._spec_rads)), self._spec_rads)
            ax.set_xticks(np.arange(len(self._res_sizes)), self._res_sizes)
            # add line at 1.0
            if 1.0 in self._spec_rads:
                line_y = self._spec_rads.index(1.0)
            else:
                ind = np.sum(np.array(self._spec_rads) < 1.0)
                line_y = ind + (1.0 - self._spec_rads[ind])/(self._spec_rads[ind+1] - self._spec_rads[ind])
            ax.axhline(y=line_y, color='blue', linestyle='--')
            

        _show_img(time_img, mean_times, 'iterations to convergence')
        _show_img(converged_plot, n_diverged, 'number of diverged runs')

        time_plot.set_ylim([0, 200])
        time_plot.set_xlim([np.min(self._spec_rads), np.max(self._spec_rads)])
        time_plot.axvline(x=1.0, color='blue', linestyle='--')
        # converged_plot.show()
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.suptitle('Mean time to fixed point on zero inputs\n(max_iter=%i, n=%i)' % (self._max_iter,self._n,))
        plt.show()


def _run_experiment(args):
    res_size, spec_rad, n_reps, max_iter = args
    times = []
    for i in range(n_reps):
        esn = EchoStateNetwork(n_input=1, n_reservoir=res_size, n_output=1,
                               spectral_radius=spec_rad, max_eq_iter=max_iter)
        times.append(esn.n_eq_washout)
    n_bad = np.sum(np.array(times) == max_iter)
    print('res_size=%i, spec_rad=%f, mean(time)=%.2f, diverged:  %i' %
          (res_size, spec_rad,  np.mean(times), n_bad))
    return (res_size, spec_rad, times)


def find_equilibria():
    ee = EquilibriumExplorer()
    ee.plot_results()


if __name__ == "__main__":
    find_equilibria()
