import numpy as np
import matplotlib.pyplot as plt
from mpl_plots import project_binary_clustering, plot_binary_clustering
from util import pca
from matplotlib.widgets import Button
import logging
import pickle


def plot_pairwise_digit_confusion(ax, results, which='test'):
    """
    show an image where the i,j value is the accuracy distinguishing between digits i and j.
    :param ax: matplotlib axis
    :param results: list of MNISTResult objects
    """
    conf_img = np.zeros((10, 10), dtype=np.float32)
    for result in results:
        pair = result.digits
        conf_img[pair[0], pair[1]] = result.accuracy[which]
        conf_img[pair[1], pair[0]] = result.accuracy[which]
    ret = ax.imshow(conf_img, cmap='hot', interpolation='nearest')
    # show all axis ticks
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    return ret


def plot_full_confusion(ax, results, **kwargs):
    """
    :param results:  list of MNISTResult objects, bootsrap samples, 
        each result has a confusion matrix, since each trial is a 10-way classification
        so plot the average confusion matrix (and colorbar) in the axes.
    """
    conf_mat = np.zeros((10, 10))
    for res in results:
        conf_mat += res._get_confusion_matrix(**kwargs)
    conf_mat /= len(results)
    img = ax.imshow(conf_mat, cmap='hot', interpolation='nearest')
    # show all axis ticks
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    return img


def plot_pairwise_clusterings(ax, results, data, which='test'):
    """
    Plot the clustering results for each digit pair.
    :param ax: matplotlib axis
    :param results: list of MNISTResult objects
    :param data: MNISTData object, for looking the original data
    """
    v_digits = []
    h_digits = []

    for result in results:
        true_labels = result.true_labels[which]
        pred_labels = result.pred_labels[which]
        true_labels, pred_labels = _relabel_binary(true_labels, pred_labels)
        pair = result.digits

        if which == 'train':
            data0 = data.train[pair[0]][result.inds[which][pair[0]]]
            data1 = data.train[pair[1]][result.inds[which][pair[1]]]
        else:
            data0 = data.test[pair[0]][result.inds[which][pair[0]]]
            data1 = data.test[pair[1]][result.inds[which][pair[1]]]
        all_data = np.vstack((data0, data1))

        unit_points = project_binary_clustering(all_data, pred_labels)

        points_shifted = unit_points + np.array(pair)

        good, bad = plot_binary_clustering(ax, points_shifted, pred_labels.astype(int), true_labels.astype(int),
                                           point_size=2, circle_size=15)
        v_digits.append(pair[1])
        h_digits.append(pair[0])

    # add legend
    good_size = good.get_sizes()[0]
    bad_size = bad.get_sizes()[0]
    c1_good_h = ax.scatter([], [], s=good_size, c='blue', label='Correct A')
    c0_good_h = ax.scatter([], [], s=good_size, c='orange', label='Correct B')
    c1_bad_h = ax.scatter([], [], s=bad_size, edgecolors='blue', label='Should be A', facecolors='none')
    c0_bad_h = ax.scatter([], [], s=bad_size, edgecolors='orange', label='Should be B', facecolors='none')
    ax.legend(handles=[c1_good_h, c0_good_h, c1_bad_h, c0_bad_h], loc='lower right')

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


def _relabel_binary(true, pred):
    labels = sorted(np.unique(true))
    true_bin = (true == labels[1]).astype(int)
    pred_bin = (pred == labels[1]).astype(int)
    return true_bin, pred_bin


def _get_failed_pair_img(result, data, max_per_pair=50, which='test', invert=True):
    """
    Show the misclassifications.
    :param result: MNISTResult object from a pairwise clustering
    :param data: MNISTData data object
    """
    pair = result.digits
    true_labels = result.true_labels[which]
    pred_labels = result.pred_labels[which]
    a_lab, b_lab = (true_labels == pair[0]), (true_labels == pair[1])

    # get the indices of the misclassified points
    # need to separate these by digit since sample indices are by digit
    # by convention, the lower digit's data is before the higher digit.
    bad_label_inds = {0: np.where(a_lab & (pred_labels == pair[1]))[0],
                      1: np.where(b_lab & (pred_labels == pair[0]))[0] - np.sum(a_lab)}

    print("Found %i bad %i's and %i bad %i's." % (len(bad_label_inds[0]), pair[0], len(bad_label_inds[1]), pair[1]))
    n_a_bad = min(len(bad_label_inds[0]), max_per_pair//2)
    n_b_bad = min(len(bad_label_inds[1]), max_per_pair//2)

    # Fit all images in a square
    square_size =np.ceil(np.sqrt(n_a_bad+ n_b_bad)).astype(int)
    n_col, n_row = square_size, square_size
    img = np.zeros((28*n_row, 28*n_col), dtype=np.uint8)
    col, row = 0, 0
    # class 0 errors start on left
    data_src = data.test if which == 'test' else data.train
    ind_src = result.inds['test'] if which == 'test' else result.inds['train']

    for i in range(n_a_bad):
        #
        digit_img = data_src[pair[0]][ind_src[pair[0]][bad_label_inds[0][i]]].reshape(28, 28)
        img[row*28:(row+1)*28, col*28:(col+1)*28] = digit_img
        row += 1
        if row == n_row:
            row = 0
            col += 1

    div_col, div_row = col,row

    for i in range(n_b_bad):
        digit_img = data_src[pair[1]][ind_src[pair[1]][bad_label_inds[1][i]]].reshape(28, 28)
        img[row*28:(row+1)*28, col*28:(col+1)*28] = digit_img
        row += 1
        if row == n_row:
            row = 0
            col += 1

    # draw a line between good and bad
    print("Size: %i x %i (img %i x %i)" % (n_row, n_col, img.shape[0], img.shape[1]))
    for r in range(n_row):
        # vertical line
        col_offset = 1 if r < div_row else 0
        print(col_offset)
        x_span = (div_col + col_offset ) * 28 , (div_col + col_offset ) * 28+1
        y_span = max(0,r * 28), min(img.shape[0],(r+1) * 28)
        print("v(r-%i, c=%i): (row %i) x: %s,  y: %s"%(div_row, col,r,x_span, y_span))
        img[y_span[0]: y_span[1], x_span[0]: x_span[1]] = 255
        
    if div_row >0:
        # horizontal 
        x_span = div_col*28,(div_col+1)*28
        y_span=(div_row)*28-1,(div_row)*28
        print("H(r-%i, c=%i): x: %s,  y: %s"%(div_row, div_col, x_span, y_span))
        img[y_span[0]: y_span[1], x_span[0]: x_span[1]] = 255


    if invert:
        img = 255 - img
    return img


def test_get_failed_pair_img():
    from mnist_data import MNISTData
    from mnist_common import make_fake_pairwise_result
    data = MNISTData(pca_dim=30)
    sample = data.get_sample(500, 500, digits=(5, 8))
    
    for p,n_wrong in enumerate( [0, 1, 5, 10, 20, 40, 50, 100, 150]):
        result = make_fake_pairwise_result(sample, (5, 8), n_wrong)
        img = _get_failed_pair_img(result, data)
        plt.subplot(3,3, p+1)
        plt.imshow(img, cmap='gray')
        # axis box off
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    


def show_failed_pairs(ax, results, data, max_per_pair=100, which='test'):
    """
    For each pair of digits, create an image showing each digit that was misclassified.
    Put the false pair[0] digits on the left, false pair[0] digits on the right.
    Show each image in a 9x8 grid, as the pairwise clusterings.
    """
    raise Exception("Not written yet.")


def plot_extreme_pairs(results, data, n=3, title=None, which='test'):
    """
    Show on n rows the best & worst clusterings (digit images in 2d embeddings).
    :param results: list of MNISTResult objects, one per digit pair 
    :param data: MNISTData object, for looking up the original data
    :param n: number of pairs to show
    :param title: title prefix for the figures
    :param which: 'train' or 'test' results
    returns: figure & axis array objects
    """
    fig, axes = plt.subplots(n, 4, figsize=(7, 2*n))
    if n == 1:
        axes = np.array([axes])

    def _add_pair(ax, result):
        # results don't contain original data, so we have to look it up and re-project it
        pair = result.digits
        inds = result.inds[which]

        if which == 'test':
            x0, x1 = data.test[pair[0]][inds[pair[0]]], data.test[pair[1]][inds[pair[1]]]
        else:
            x0, x1 = data.train[pair[0]][inds[pair[0]]], data.train[pair[1]][inds[pair[1]]]

        images0 = x0.reshape(-1, 28, 28)
        images1 = x1.reshape(-1, 28, 28)
        x0 = result.pca_transf.transform(x0)
        x1 = result.pca_transf.transform(x1)
        points = np.vstack((x0, x1))

        true_labels = np.hstack((np.ones(len(inds[pair[0]]))*pair[0],
                                 np.ones(len(inds[pair[1]]))*pair[1]))
        pred_labels = result.pred_labels[which]
        true_labels, pred_labels = _relabel_binary(true_labels, pred_labels)

        task_title = "%i vs %i" % (pair[0], pair[1])

        show_digit_cluster_collage_binary(ax,
                                          np.vstack((images0, images1)),
                                          points,
                                          pred_labels,
                                          true_labels,
                                          max_n_imgs=300,
                                          title=task_title)

    best = sorted(results, key=lambda res: res.accuracy[which], reverse=True)[:n]
    worst = sorted(results, key=lambda res: res.accuracy[which])[:n][::-1]

    for i, res in enumerate(best):
        _add_pair((axes[i][0], axes[i][1]), res)

    for i, res in enumerate(worst):
        _add_pair((axes[i][2], axes[i][3]), res)

    _title = "%i best and worst cluster/digit correspondences\n(left and right, respectively), %s data" % (n, which)
    if title is not None:
        _title = "%s: %s" % (title, _title)
    fig.suptitle(_title)
    return fig, axes


def plot_full_embedding(results, data, title, which='train', **kwargs):
    """
    Plot the full embedding of all digits, color-coded by the best result.
    :param ax: matplotlib axis
    :param results: list of MNISTResult objects
    :param data: MNISTData object, for looking up the original data
    """
    fig = plt.figure(figsize=(6, 6))
    ax_data = fig.add_subplot()
    fig.subplots_adjust(right=.8)

    best_result = max(results, key=lambda res: res.accuracy[which])

    best_labels = best_result.pred_labels[which]
    # true_labels = best_result.true_labels[which]
    indices = best_result.inds[which]
    if which == 'train':
        all_data = np.vstack([data.train[i][indices[i]] for i in range(10)])
    else:
        all_data = np.vstack([data.test[i][indices[i]] for i in range(10)])
    all_imgs = all_data.reshape(-1, 28, 28)

    artists, colors = show_digit_cluster_collage_full(ax_data, all_imgs, all_data, best_labels, **kwargs)
    # extract colors for each digit

    # Add 10 buttons on the side of the image to turn off/on each digit
    digits_active = {i: True for i in range(10)}

    def _toggle_activity(digit):
        digits_active[digit] = not digits_active[digit]
        state = digits_active[digit]
        logging.info("Setting digit %i to %s." % (digit, state))
        for artist in artists[digit]:
            artist.set_visible(state)
        plt.draw()

    all_active = [True]  # mutable object to use

    def _toggle_all():
        all_active[0] = not all_active[0]
        logging.info("Setting all digits to %s." % all_active[0])
        for i in range(10):
            if digits_active[i] != all_active[0]:
                _toggle_activity(i)
        plt.draw()

    button_height = 0.05
    button_width = 0.07
    button_spacing = 0.01
    y = 0.2
    global buttons  # to prevent garbage collection
    buttons = []
    for i in range(10):
        button_ax = plt.axes([.83, y, button_width, button_height])
        button = Button(button_ax, "%i" % i, color=colors[i])
        button.on_clicked(lambda event, i=i: _toggle_activity(i))
        y += button_height + button_spacing
        buttons.append(button)
    y += button_spacing
    button_ax = plt.axes([.83, y, button_width, button_height])
    button = Button(button_ax, "All")
    button.on_clicked(lambda event: _toggle_all())
    buttons.append(button)
    plt.suptitle(title)
    # turn off ticks
    ax_data.set_xticks([])
    ax_data.set_yticks([])


def show_digit_cluster_collage_full(ax, images, points, true_labels, max_n_imgs=100, image_extent_frac=0.01):
    """
    Create a 2d embedding with PCA on the cluster centers.
    Plot max_n_imgs of each digit in its embedded location.

    :param ax: matplotlib axis
    :param images: N x 28 x 28 array of images
    :param points: N x D array of points
    :param true_labels: N array of integers in [0, 9], ground truth labels
    :param max_n_imgs: integer, max number of images to plot in each axis.
    :param image_extent_frac: float, fraction of the image width to extend the extent of the image
    :returns: dict of lists of matplotlib artists
    """
    colors_rgb = np.array([[0, 0, 0],
                           [1, 0, 0],
                           [0, 1, 0],
                           [0, 0, 1],
                           [1, 1, 0],
                           [1, 0, 1],
                           [0, 1, 1],
                           [1, 0.5, 0],
                           [0.5, 0, 1],
                           [0, 0.5, 1]], dtype=np.float32)
    centers = [np.mean(points[true_labels == i], axis=0) for i in range(10)]
    dims, _ = pca(np.array(centers), 2)
    points = np.dot(points, dims)
    x_span = np.max(points[:, 0]) - np.min(points[:, 0])
    side = x_span * image_extent_frac
    artists = {}
    for digit in range(10):
        artists[digit] = []
        digit_inds = np.where(true_labels == digit)[0]
        n_imgs = min(max_n_imgs, len(digit_inds))
        sample = np.random.choice(digit_inds, n_imgs, replace=False)
        imgs = colorize(images[sample], colors_rgb[digit])
        pts = points[sample]
        for i in range(n_imgs):
            art = ax.imshow(imgs[i], extent=(pts[i, 0]-side, pts[i, 0]+side,
                                             pts[i, 1]-side, pts[i, 1]+side))
            artists[digit].append(art)
    ax.set_xlim(np.min(points[:, 0])-side, np.max(points[:, 0])+side)
    ax.set_ylim(np.min(points[:, 1])-side, np.max(points[:, 1])+side)
    return artists, colors_rgb


def colorize(imgs, color):
    """
    Colorize a set of images with a single color.
    :param imgs: N x 28 x 28 array of images
    :param color: 4-tuple of floats in [0, 1]
    :return: N x 28 x 28 x 4 array of colorized images
    """
    imgs = np.repeat(imgs[..., np.newaxis], 4, axis=3).astype(np.float32)
    imgs[:, :, :, :3] = imgs[:, :, :, :3] / 255.0 * color[:3]
    imgs[:, :, :, 3] /= 255.
    return imgs


def show_digit_cluster_collage_binary(ax, images, points, pred_labels, true_labels, max_n_imgs=200, image_extent_frac=0.03, invert=True, title=None):
    """
    Plot the images in their embedded locations to show the clustering results.
    Make 2 plots w/correct predictions on the left, errors on the right.

    :param ax: list of 2 matplotlib axis objects
    :param images: N x 28 x 28 array of images
    :param points: N x D array of points
    :param pred_labels: N array of integers in [0, 1], cluster assignments
    :param true_labels: N array of integers in [0, 1], ground truth labels
    :param max_n_imgs: integer, max number of images to plot in each axis.
    :param image_extent_frac: float, fraction of the image width to extend the extent of the image
    :param invert: True for plotting Black characters on white background
    """
    # flatten to 2d
    points = project_binary_clustering(points, pred_labels)
    x_span = np.max(points[:, 0]) - np.min(points[:, 0])
    side = x_span * image_extent_frac
    # do correct side on ax[0]
    correct = pred_labels == true_labels
    n_correct = np.sum(correct)
    if n_correct > max_n_imgs:
        idx = np.random.choice(np.where(correct)[0], max_n_imgs, replace=False)
    else:
        idx = np.where(correct)[0]
    for i in idx:
        img = images[i] if not invert else 255 - images[i]
        ax[0].imshow(img, extent=(points[i, 0]-side, points[i, 0]+side,
                                  points[i, 1]-side, points[i, 1]+side), cmap='gray',
                     alpha=images[i]/255.)
    ax[0].set_xlim(-side, 1.+side)
    ax[0].set_ylim(-side, 1.+side)
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    # do errors side on ax[1]
    n_errors = np.sum(~correct)
    if n_errors > max_n_imgs:
        idx = np.random.choice(np.where(~correct)[0], max_n_imgs, replace=False)
    else:
        idx = np.where(~correct)[0]
    for i in idx:
        img = images[i] if not invert else 255 - images[i]
        ax[1].imshow(img, extent=(points[i, 0]-side, points[i, 0]+side,
                                  points[i, 1]-side, points[i, 1]+side), cmap='gray',
                     alpha=images[i]/255.)
    ax[1].set_xlim(-side, 1.+side)
    ax[1].set_ylim(-side, 1.+side)
    accuracy = np.mean(correct)
    c_title = "Correct predictions" if title is None else "Correct: %s" % title
    e_title = "Errors" if title is None else "Errors: %s" % title
    ax[0].set_xlabel(c_title)
    ax[1].set_xlabel(e_title)
    ax[1].set_xticks([])
    ax[1].set_yticks([])
    ax[0].set_ylabel("Accuracy: %.3f" % accuracy)


def plot_pairwise_accuracy_boxplot(ax, results, title, which='test'):
    """
    Plot the pairwise accuracy as a boxplot.
    :param ax: matplotlib axis
    :param results: list of MNISTResult objects
    """
    accs = [res.get_info('accuracies')[which] for res in results]
    pairs = [res.digits for res in results]
    ax.boxplot(accs)
    ax.set_xticklabels(["%i - %i" % (pair[0], pair[1]) for pair in pairs], rotation=90)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.grid(axis='both')


#######################################################
# Testing functions
#######################################################

def test_show_digit_cluster_collage_binary():
    # making test data
    from mnist_data import MNISTData
    n_train = 500
    data = MNISTData(pca_dim=30)
    sample = data.get_sample(n_train, n_train)

    # show 5s and 8s, make 4 5s wrong and 30 8s wrong
    pair = 5, 8

    images0 = data.train[pair[0]][sample.train_inds[pair[0]]].reshape(-1, 28, 28)
    images1 = data.train[pair[1]][sample.train_inds[pair[1]]].reshape(-1, 28, 28)

    points_pca = np.vstack((sample.train[pair[0]], sample.train[pair[1]]))
    images_both = np.vstack((images0, images1))

    train_lab = np.concatenate((np.ones(n_train) * pair[0],
                                np.ones(n_train)*pair[1])).astype(int)
    pred_lab = train_lab.copy()
    pred_lab[:4] = pair[1]
    pred_lab[-30:] = pair[0]

    # plotting
    fig, ax = plt.subplots(1, 2)
    show_digit_cluster_collage_binary(ax, images_both, points_pca, pred_lab, train_lab)
    plt.suptitle("Projected digits, errors\n circled w/correct colors.")
    plt.tight_layout()
    plt.show()


def test_plot_full_embedding():
    # making test data, a fake result
    from mnist_data import MNISTDataPCA
    from mnist_common import MNISTResult
    data = MNISTDataPCA(dim=30)
    n_imgs = 40
    inds = {i: data.get_digit_sample_inds(i, n_imgs) for i in range(10)}
    all_data = np.vstack([data.get_digit(i)[inds[i]] for i in range(10)])
    true_labels = np.concatenate([i*np.ones(n_imgs) for i in range(10)])
    # cluster with K-means(10)
    from clustering import KMeansAlgorithm
    ka = KMeansAlgorithm(10)
    ka.fit(all_data)
    result = MNISTResult(ka, all_data, true_labels, inds)
    results = [result]
    plot_full_embedding(results, data, image_extent_frac=0.03)


def test_plot_extreme_pairs():
    with open('KMeans_pairwise_r=15_n=1000.pkl', 'rb') as f:
        results = pickle.load(f)
    from mnist_data import MNISTDataPCA
    data = MNISTDataPCA(dim=30)
    plot_extreme_pairs(results, data, n=3)

    plt.show()


def test_plot_full_confusion():
    file = 'KMeans_full_r=15_n=1000_b=100.pkl'
    with open(file, 'rb') as f:
        results = pickle.load(f)
    fig, ax = plt.subplots(1, 1)
    img = plot_full_confusion(ax, results)
    fig.colorbar(img, ax=ax)
    plt.show()


if __name__ == '__main__':
    # show_clusters(np.random.randn(100, 2), np.random.randint(0, 10, 100))

    # test_plot_full_embedding()
    # test_plot_extreme_pairs()
    # test_plot_full_confusion()
    # test_show_digit_cluster_collage_binary()
    # plt.show()
    test_get_failed_pair_img()
