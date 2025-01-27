import numpy as np
import matplotlib.pyplot as plt
from mpl_plots import project_binary_clustering, plot_binary_clustering, show_digit_cluster_collage


def plot_pairwise_digit_confusion(ax, results):
    """
    Plot the pairwise digit confusion matrix.
    :param ax: matplotlib axis
    :param results: list of MNISTResult objects
    """
    conf_img = np.zeros((10, 10), dtype=np.float32)
    for result in results:
        conf_img[int(result.aux['pair'][0]), int(result.aux['pair'][1])] = result.accuracy
        conf_img[int(result.aux['pair'][1]), int(result.aux['pair'][0])] = result.accuracy
    ret = ax.imshow(conf_img, cmap='hot', interpolation='nearest')
    # show all axis ticks
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    return ret


def plot_pairwise_clusterings(ax, results, data):
    """
    Plot the clustering results for each digit pair.
    :param ax: matplotlib axis
    :param results: list of MNISTResult objects
    :param data: MNISTData object, for looking the original data
    """
    v_digits = []
    h_digits = []

    for result in results:
        pred_labels = result.pred_labels
        true_labels = result.true_labels
        pair = result.aux['pair']
        data0 = data.get_digit(pair[0])[result.inds[0]]
        data1 = data.get_digit(pair[1])[result.inds[1]]
        all_data = np.vstack((data0, data1))
        unit_points = project_binary_clustering(all_data, pred_labels)
        points_shifted = unit_points + np.array(pair)

        plot_binary_clustering(ax, points_shifted, pred_labels.astype(int), true_labels.astype(int),
                               point_size=2, circle_size=15)
        v_digits.append(pair[1])
        h_digits.append(pair[0])
    v_digits = np.sort(np.unique(v_digits))
    h_digits = np.sort(np.unique(h_digits))
    # draw black lines between pairs
    x_lim, y_lim = [0, 9], [1, 10]
    for i in range(9):
        if i < 9:
            ax.plot([i, i], y_lim, color='black', linewidth=0.5)
        ax.plot(x_lim, [i+1, i+1], color='black', linewidth=0.5)
    # set x-ticks and y-ticks for all integers
    ax.set_xticks(np.array(h_digits)+.5)
    ax.set_yticks(np.array(v_digits)+.5)
    ax.set_xticklabels(["%i" % i for i in h_digits])
    ax.set_yticklabels(["%i" % i for i in v_digits])

    ax.xaxis.tick_top()
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)


def plot_best_and_worst_pairs(results, data, n=2, title=""):
    """
    Show (each in a new figure) the best and worst clusterings (digit images in 2d embeddings).
    :param results: list of MNISTResult objects
    :param data: MNISTData object, for looking up the original data
    :param n: number of best and worst pairs to show
    :param title: title prefix for the figures
    """

    def _show_pair(fig, ax, result, title_2, index):
        pair = result.aux['pair']
        pair = result.aux['pair']
        data0 = data.get_digit(pair[0])[result.inds[0]]
        data1 = data.get_digit(pair[1])[result.inds[1]]
        img0 = data.get_images(pair[0])[result.inds[0]]
        img1 = data.get_images(pair[1])[result.inds[1]]
        all_data = np.vstack((data0, data1))
        all_imgs = np.vstack((img0, img1))
        show_digit_cluster_collage(ax,
                                   all_imgs,
                                   all_data,
                                   result.pred_labels,
                                   result.true_labels,
                                   max_n_imgs=300)
        fig.suptitle("%s - %s pairs, showing %i of %i:  %s" % (title, title_2, index+1, n, pair))

    best = sorted(results, key=lambda res: res.accuracy, reverse=True)[:n]
    worst = sorted(results, key=lambda res: res.accuracy)[:n]

    for i, res in enumerate(best):
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        _show_pair(fig, ax, res, "Best", i)
        plt.show()

    for i, res in enumerate(worst):
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        _show_pair(fig, ax, res, "Worst", i)
        plt.show()
