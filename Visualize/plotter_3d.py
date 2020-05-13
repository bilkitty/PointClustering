import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os.path as path
import pickle


C_MAP = "Spectral"


def plot_curve(bezier_curve, ctrl_points, filepath=""):
    assert(bezier_curve.shape[1] == 3), f"expected 3-d bezier, got: {bezier_curve}"
    assert(ctrl_points.shape[1] == 3), f"expected 3-d points, got: {ctrl_points}"

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot Bezier curve
    X = bezier_curve[:, 0]
    Y = bezier_curve[:, 1]
    Z = bezier_curve[:, 2]
    ax.plot(X, Y, Z, markersize=5)

    # Plot control points
    ax.scatter(ctrl_points[:, 0], ctrl_points[:, 1], ctrl_points[:, 2], c="red")
    for nr in range(len(ctrl_points)):
        ax.text(ctrl_points[nr][0], ctrl_points[nr][1], ctrl_points[nr][2], s=str(nr))

    ax.set_xlabel("x")
    ax.set_ylabel("y")

    if filepath is "":
        plt.show()
    else:
        # Ordinary save for quick peek - not interactive
        plt.savefig(filepath)
        # Save inputs to regenerate interactive figure later
        pickle_file = filepath.split(".")[0] + ".pickle"
        inputs = {
           "bezier_curve" : bezier_curve,
           "ctrl_points" : ctrl_points
        }

        with open(pickle_file, 'wb') as f:
            pickle.dump(inputs, f)
        print(f"saved inputs to {pickle_file}")


def plot_clusters(clusters, centers, filepath="", key_points=None):
    assert(clusters[1].shape[0] == 3), f"expected 3-d cluster points, got: {clusters[1].shape}"
    assert(centers.shape[0] == 3), f"expected 3-d centers, got: {centers.shape}"

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for k in range(len(clusters.keys())):
        idx = k + 1
        # Plot clusters
        if clusters[idx].shape[1] != 0:
            ax.scatter(clusters[idx][0, :],
                       clusters[idx][1, :],
                       clusters[idx][2, :],
                       cmap=C_MAP,
                       s=5)

    # Plot centers
    ax.scatter(centers[0, :], centers[1, :], centers[2, :], s=30, marker="s", c="black")
    ax.scatter(key_points[0, :], key_points[1, :], key_points[2, :], s=5, marker="o", c="red")
    plt.xlabel("x")
    plt.ylabel("y")

    if filepath is "":
        plt.show()
    else:
        # Ordinary save for quick peek - not interactive
        plt.savefig(filepath)
        # Save inputs to regenerate interactive figure later
        pickle_file = filepath.split(".")[0] + ".pickle"
        inputs = {
           "clusters" : clusters,
           "centers" : centers,
           "key_points" : key_points,
        }

        with open(pickle_file, 'wb') as f:
            pickle.dump(inputs, f)
        print(f"saved inputs to {pickle_file}")

    plt.clf()
    plt.close()


def plot_hierarchical_clusters(points, clusters, key_points=None, filepath=""):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Plot clusters
    ax.scatter(points[:, 0],
               points[:, 1],
               points[:, 2],
               c=clusters,
               cmap=C_MAP,
               s=5)

    ax.scatter(key_points[0, :], key_points[1, :], key_points[2, :], s=5, marker="o", c="red")
    plt.xlabel("x")
    plt.ylabel("y")

    todo_save = filepath is not ""
    if not todo_save:
        plt.show()
        u_response = input("Save this figure? (Y/N/Q)")
        todo_save = u_response.lower() == 'y'

    if todo_save:
        u_response = input("Where? (directory path)")
        if filepath is "":
            filepath = path.join(u_response, "hierarchical_clusters_plot.png")
        # Ordinary save for quick peek - not interactive
        plt.savefig(filepath)
        # Save inputs to regenerate interactive figure later
        pickle_file = filepath.split(".")[0] + ".pickle"
        inputs = {
           "clusters" : clusters,
           "points" : points,
           "key_points" : key_points,
        }
        with open(pickle_file, 'wb') as f:
            pickle.dump(inputs, f)
        print(f"saved inputs to {pickle_file}")

    plt.clf()
    plt.close()
