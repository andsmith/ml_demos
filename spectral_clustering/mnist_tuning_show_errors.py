"""
Load results from MNIST tuning.
Plot accuracy curves.
Provide UI where user selects parameter value (x), and a curve (y), and clicks to show
the results for clusterings at that value.
"""
from mnist_tuning import MNISTPairwiseTuner
import matplotlib.pyplot as plt
import numpy as np
import logging


class PairwiseTunerErrGallery(MNISTPairwiseTuner):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.run()  # load results

        self._mouseover = {'graph_name': None,
                           'axis_ind': None,  # left or right
                           'xy': None,
                           'param_ind': None}
        self._ax = None

    def run_ui(self):
        self._fig, self._ax, self._values = self._plot_accuracy_curves('test')
        self._vlines = []
        for ind in range(2):
            print("Drawing line on axis %i" % ind)
            vline = self._ax[ind].axvline(0, linestyle=':', color='black', alpha=1)
            # make invisible
            # vline.set_visible(False)
            self._vlines.append(vline)

        self._fig.canvas.mpl_connect('button_press_event', self._onclick)
        self._fig.canvas.mpl_connect('motion_notify_event', self._onhover)
        plt.show()

    def _is_tool_active(self):
        return self._fig.canvas.toolbar.mode != ""

    def _onclick(self, event):
        if event.inaxes is None or self._is_tool_active():
            return
        print('button_press_event', event)

    def _get_mouseover(self, event):
        """
        Figure out which curve is closest to the mouse and where it's closest point is.
        :param x: x value of mouse
        :param y: y value of mouse

        """
        if event.inaxes is None or self._is_tool_active():
            return None
        ax_ind = int(event.inaxes == self._ax[1])
        best = {'_y_diff': np.inf,
                'graph_name': None,
                'param_index': None}
        x, y = event.xdata, event.ydata
        for g_name in self._values:
            if self._values[g_name]['plot_side'] != ax_ind:
                continue
            diffs = np.abs(self._values[g_name]['param_vals'] - x)
            param_index = np.argmin(diffs)
            mean_acc = self._values[g_name]['mean_acc'][param_index]
            y_diff = np.abs(mean_acc - y)
            if y_diff < best['_y_diff']:
                best['_y_diff'] = y_diff
                best['graph_name'] = g_name
                best['param_index'] = param_index
        return {'graph_name': best['graph_name'],
                'axis_ind': ax_ind,
                'xy': (x, y),
                'param_ind': best['param_index']}

    def _refresh(self, event):
        # update lines
        if event.inaxes is None or self._is_tool_active():
            [vline.set_visible(False) for vline in self._vlines]
        else:
            ax_ind = int(event.inaxes == self._ax[1])
            x = event.xdata
            print("Setting vline %i to %s" % (ax_ind, x))
            self._vlines[1-ax_ind].set_visible(False)
            self._vlines[ax_ind].set_visible(True)
            self._vlines[ax_ind].set_xdata([x, x])
        # update title
        if self._mouseover is not None:
            param_ind = self._mouseover['param_ind']
            param_val = self._values[self._mouseover['graph_name']]['param_vals'][param_ind]
            accuracy = self._values[self._mouseover['graph_name']]['mean_acc'][param_ind]
            self._fig.suptitle("Selected: %s (param=%.1f, acc=%.4f)" % (self._mouseover['graph_name'], param_val, accuracy))
        else:
            self._fig.suptitle("")
        self._fig.canvas.draw()

    def _get_results(self, g_name, p_ind):
        """
        # get all the results (digit pairs) for a given parameter value
        """
        result_list = [result_pair[p_ind] for result_pair in self._results[g_name]]
        print(len(result_list))
        return result_list

    def _onhover(self, event):
        self._mouseover = self._get_mouseover(event)
        self._refresh(event)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    PairwiseTunerErrGallery().run_ui()
