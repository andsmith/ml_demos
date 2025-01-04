"""
Show a plot with the 4 kinds of test data in make_data.py
"""
import matplotlib.pyplot as plt
from make_data import make_minimal_data, make_spiral_data, make_checker_data, make_bump
import numpy as np
from plotting import plot_dataset


def show_datasets():
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    n_points = 40
    
    datasets = {'minimal': make_minimal_data(),
                'spiral': make_spiral_data(n_points)[:2],
                'checker': make_checker_data(n_points),
                'bump': make_bump(n_points, 0.2, 0.33, 0.0, 0.0)}
    for ax, dataset in zip(axs.ravel(), datasets.keys()):
        print("PLOTTING", dataset, "n_pts", len(datasets[dataset][0]))
        print(type(dataset[0]), len(dataset))
        X, y = datasets[dataset]
        plot_dataset(ax, X, y)
        ax.set_title("Dataset: " + dataset)
        ax.set_aspect('equal')
        ax.yaxis.set_visible(False)
        ax.xaxis.set_visible(False)
    #plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    show_datasets()
