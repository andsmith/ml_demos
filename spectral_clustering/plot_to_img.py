"""
Plot with matplotlib but to an image.
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

MONITOR_DPI = 100

class PlotRenderer(object):
    """
    Draw matplotlib plots to an exact image size.
    Usage:
        plot = PlotRenderer((200, 200), 'Test Plot')
        fig, ax = plot.get_axis()
        ax.plot ...
        img = plot.render_fig(fig)    
    """
    def __init__(self, size, *args, **kwargs):
        """
        :param size: (width, height) of the images to create
        :param title: title of the plot
        :param args: arguments to pass to the plot
        :param kwargs: keyword arguments to pass to the plot

        """
        self._size = size
        self._values = None
        self._visible = True
        self._disp_img = None
        self._args, self._kwargs = args, kwargs
        print("PlotRenderer: size={}, args={}, kwargs={}".format(size, args, kwargs))

    def _get_fig_size(self):
        """
        Get the width, height and dpi of the figure so it matches the required size.
        """
        width, height = self._size
        fig_width = width / MONITOR_DPI
        fig_height = height / MONITOR_DPI
        return fig_width, fig_height
    
    def get_axis(self):
        fig, ax= plt.subplots(figsize=self._get_fig_size(), dpi=MONITOR_DPI)
        return fig, ax
    
    def render_fig(self, fig, bgr=False):
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8) 
        buf.shape = (h, w, 4)
        plt.close(fig)
        buf = buf[:, :, :3]  # no alpha
        if not bgr:
            buf = buf[:, :, ::-1] 
        return buf

def _test_plot_renderer():
    plot = PlotRenderer((200, 200))
    fig, ax = plot.get_axis()
    ax.set_title('Test Plot')
    ax.plot([1, 2, 3, 4, 5])
    plt.tight_layout()
    img = plot.render_fig(fig)
    cv2.imshow('Test Plot', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':

    _test_plot_renderer()