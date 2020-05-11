import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


def plot_clusters(clusters, centers, filepath=""):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for k in range(len(clusters.keys())):
        idx = k + 1
        # Plot clusters
        if clusters[idx].shape[1] != 0:
            ax.scatter(clusters[idx][0, :],  # plot x
                        clusters[idx][1, :],  # plot y
                        cmap="coolwarm")

    # Plot centers
    ax.scatter(centers[0, :], centers[1, :], s=30, marker="x", label="centers")
    plt.xlabel("income")
    plt.ylabel("transactions")
    #plt.legend()

    if filepath is "":
        plt.show()
    else:
        plt.savefig(filepath)

    plt.clf()
