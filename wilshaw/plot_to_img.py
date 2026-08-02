"""
Render matplotlib figures to exact-size numpy RGB images, so charts can be
blitted into a tk.Label like any other image (the author's paradigm -- no
FigureCanvasTkAgg).  Adapted from spectral_clustering/plot_to_img.py, but
returns RGB by default (what PIL/ImageTk want).
"""

import matplotlib
matplotlib.use('Agg')            # headless, thread-friendly rendering backend
import matplotlib.pyplot as plt
import numpy as np

MONITOR_DPI = 100


class PlotRenderer(object):
    """
    Draw matplotlib plots to an exact pixel size.

    Usage::

        plot = PlotRenderer((640, 400))
        fig, ax = plot.get_axis()
        ax.bar(...)
        img = plot.render_fig(fig)   # (H, W, 3) uint8 RGB
    """

    def __init__(self, size):
        """:param size: (width, height) of the rendered image in pixels"""
        self._size = size

    def set_size(self, size):
        """Update the target image size (width, height)."""
        self._size = size

    def _fig_size(self):
        w, h = self._size
        return w / MONITOR_DPI, h / MONITOR_DPI

    def get_axis(self, n_rows=1, n_cols=1, sharex=False, sharey=False):
        """Create a figure/axes sized to match the target image."""
        fig, ax = plt.subplots(n_rows, n_cols, figsize=self._fig_size(),
                               dpi=MONITOR_DPI, sharex=sharex, sharey=sharey)
        return fig, ax

    def render_fig(self, fig):
        """
        Rasterise ``fig`` and return it as an (H, W, 3) uint8 RGB array.
        Closes the figure to avoid leaks.
        """
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).copy()
        buf.shape = (h, w, 4)
        plt.close(fig)
        return buf[:, :, :3]          # drop alpha; already RGB


def _test_plot_renderer():
    import cv2
    plot = PlotRenderer((400, 300))
    fig, ax = plot.get_axis()
    ax.set_title('Test Plot')
    ax.bar(['a', 'b', 'c'], [3, 1, 4])
    fig.tight_layout()
    img = plot.render_fig(plot and fig)
    assert img.shape[2] == 3
    cv2.imshow('Test Plot', img[:, :, ::-1])   # RGB -> BGR for cv2
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    _test_plot_renderer()
