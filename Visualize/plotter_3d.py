import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os.path as path
import pickle


def plot_curve(bezier_curve, ctrl_points, filepath=""):
    assert(bezier_curve.shape[1] == 3), f"expected 3-d bezier, got: {bezier_curve}"
    assert(ctrl_points.shape[1] == 3), f"expected 3-d points, got: {ctrl_points}"

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot Bezier curve
    X = bezier_curve[:, 0]
    Y = bezier_curve[:, 1]
    Z = bezier_curve[:, 2]
    ax.plot(X, Y, Z)
    #from matplotlib import cm
    #cset = ax.contour(X, Y, Z, zdir='z', offset=np.min(Z), cmap=cm.coolwarm)
    #cset = ax.contour(X, Y, Z, zdir='x', offset=np.min(X), cmap=cm.coolwarm)
    #cset = ax.contour(X, Y, Z, zdir='y', offset=np.min(Y), cmap=cm.coolwarm)

    # Plot control points
    ax.scatter(ctrl_points[:, 0], ctrl_points[:, 1], ctrl_points[:, 2], c="red")
    for nr in range(len(ctrl_points)):
        ax.text(ctrl_points[nr][0], ctrl_points[nr][1], ctrl_points[nr][2], s=str(nr))

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    if filepath is "":
        plt.show()
    else:
        plt.savefig(filepath)
        pickle_file = open(filepath.split(".")[0] + ".pickle", 'wb')
        pickle.dump(plt.gcf(), pickle_file)
        pickle_file.close()


def plot_clusters(clusters, centers, filepath=""):
    assert(clusters[1].shape[0] == 3), f"expected 3-d cluster points, got: {clusters[1].shape}"
    assert(centers.shape[0] == 3), f"expected 3-d centers, got: {centers.shape}"

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for k in range(len(clusters.keys())):
        idx = k + 1
        # Plot clusters
        if clusters[idx].shape[1] != 0:
            ax.scatter(clusters[idx][0, :],  # plot x
                        clusters[idx][1, :],  # plot y
                        clusters[idx][2, :],  # plot z
                        cmap="coolwarm")

    # Plot centers
    ax.scatter(centers[0, :], centers[1, :], centers[2, :], s=30, marker="x", label="centers")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylabel("z")
    #plt.legend()

    if filepath is "":
        plt.show()
    else:
        plt.savefig(filepath)
        pickle_file = open(filepath.split(".")[0] + ".pickle", 'wb')
        pickle.dump(plt.gcf(), pickle_file)
        pickle_file.close()

    plt.clf()
    plt.close()


