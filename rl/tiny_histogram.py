"""
Draw a small histogram, part of step visualizations.
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import logging
from colors import COLOR_BG, COLOR_LINES, COLOR_TEXT


def get_n_bins(values):
    """
    Use the Freedman-Diaconis rule to determine the number of bins for a histogram.
    """
    n_distinct = len(np.unique(values))
    if n_distinct <= 5:
        return 20
    q25, q75 = np.percentile(values, [25, 75])
    iqr = q75 - q25
    bin_width = 2 * iqr / (len(values) ** (1 / 3))
    if bin_width == 0:
        bin_width = 1
    #print(values.size,n_distinct, bin_width)
    #print(values.min(), values.max(), q25, q75, iqr)

    n_bins = int(np.ceil((values.max() - values.min()) / bin_width))
    if n_bins < 1:
        return 1
    return n_bins

class TinyHistogram(object):
    """A small histogram, meant to be roughly parallel to each layer in the state representations.
       i.e. 6 in a column.
    """

    def __init__(self, size, values, color_draw=COLOR_LINES, color_text=COLOR_TEXT):
        self._size = size
        self._vals = np.array(values)
        self._color = color_draw
        self._color_text = np.array(color_text) / 255
        self._color_bg = np.array(COLOR_BG)/255


    def draw(self, img, x, y, test_plot=False, xlabel="",ylabel="", title=""):
        """
        Draw to a matplotlib axis, export to an image and paste to img.
        """
        # draw a bounding box:
        # cv2.rectangle(img, (x, y), (x + self._size[0], y + self._size[1]), self._color, 1)
        
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(self._size[0] / 100, self._size[1] / 100))
        print(fig.canvas.get_width_height()[::-1])
        ax.set_facecolor(self._color_bg)
        fig.patch.set_facecolor(self._color_bg)

        # Draw the histogram
        ax.hist(self._vals, bins=get_n_bins(self._vals), color=np.array(self._color) / 255, edgecolor=self._color_text, alpha=0.7)
        if xlabel !="":ax.set_xlabel(xlabel, fontsize=10, color=self._color_text)
        if ylabel !="": ax.set_ylabel(ylabel, fontsize=10, color=self._color_text)
        # reduce tick font label size
        ax.tick_params(axis='both', which='major', labelsize=10, color=self._color_text)
        ax.tick_params(axis='both', which='minor', labelsize=10, color=self._color_text)

        # Set limits and remove ticks
        plt.tight_layout()
        fig.subplots_adjust(bottom=.15,left=.18)
        
        if test_plot:
            plt.show()


        # Save the figure to a buffer
        fig.canvas.draw()
        img_data = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img_data = img_data.reshape(fig.canvas.get_width_height()[::-1] + (4,))

        # Paste the image into the target image
        img[y:y + self._size[1], x:x + self._size[0]] = img_data[:, :, :3]

        # Close the figure to free memory
        plt.close(fig)

class MultiHistogram(object):
    """
    Space several histograms evenly on an image.
    """
    def __init__(self, img_size, value_lists, pad=10,  **kwargs):
        """
        Divide into len(value_lists) rows, 1 column. 
        Pad on all sides & between histotrams.
        """
        v_pad_total = (len(value_lists)+1) * pad
        h_pad_total = 2 * pad
        self._v_space = (img_size[1] - v_pad_total) // len(value_lists)
        self._h_space = img_size[0] - h_pad_total
        self._pad = pad
        self._img_size = img_size
        self._tiny_size = (self._h_space, self._v_space)
        self._histograms = [TinyHistogram(self._tiny_size, values, **kwargs) for values in value_lists]

    def draw(self, img):
        """
        Draw all histograms to the image.
        """
        y = self._pad
        x = self._pad
        for l_num, hist in enumerate(self._histograms):
            hist.draw(img, x, y, ylabel="Layer %i" %(l_num+1))
            y += self._v_space + self._pad
        return img

def test_multihist():
    values = [np.random.randn(np.random.randint(1,50)**2) for _ in range(6)]
    size = (413, 980)
    mh = MultiHistogram(size, values, color_draw=COLOR_LINES, color_text=COLOR_TEXT)
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    img[:] += np.array(COLOR_BG, dtype=np.uint8)
    print("Created image of size:", img.shape, "with color", COLOR_BG)
    mh.draw(img)
    cv2.imshow("Tiny Histogram", img[:,:,::-1])
    cv2.waitKey(0)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_multihist()
