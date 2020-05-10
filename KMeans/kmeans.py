#!/usr/bin/python3
"""
src: https://medium.com/machine-learning-algorithms-from-scratch/k-means-clustering-from-scratch-in-python-1675d38eee42

Clustering algorithm summary

Given a point set P, we want to create k-clusters such that
points belonging to each cluster have higher similarity than
those belonging to other clusters. A simple example is spatial
grouping of points by a Euclidean distance threshold.

1. initialization of cluster centers
    non-trivial if we seek an efficient algo
    e.g., random init using k points in P may result in
          inefficient algo
2. compute clusters
    for each point
      compute distance to each cluster
      assign it to the nearest one
3. compute new cluster centers
    for each cluster
      compute new centroid as avg of all cluster members
4. iterate steps 2 and 3

"""
import numpy as np
import pandas as pd
import random as rand
import sys
import os
from matplotlib import pyplot as plt

MAX_ITERATIONS = 20
K_CLUSTERS = 5
FIGURES_DIR = "/home/bilkit/Workspace/PointClustering/KMeans/results"
UNIQUE_ID = rand.randint(0, 100000)


def load_data(filepath):
    assert(os.path.exists(filepath)), f"'{filepath}' does not exist"
    return pd.read_csv(filepath)


def create_point_set(csv_data, c1, c2):
    assert(0 <= c1 < csv_data.shape[1]), f"{c1} is out of bounds [0,{csv_data.shape[1]}]"
    assert(0 <= c2 < csv_data.shape[1]), f"{c2} is out of bounds [0,{csv_data.shape[1]}]"
    return csv_data.iloc[:, [c1, c2]].values


def plot_clusters(itr, clusters, centers):
    if itr % 2 != 0:
        return

    colors = ['red', 'blue', 'green', 'cyan', 'magenta']
    labels=['c1', 'c2', 'c3', 'c4', 'c5']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for k in range(len(clusters.keys())):
        idx = k + 1
        # Plot clusters
        if clusters[idx].shape[1] != 0:
            ax.scatter(clusters[idx][0, :],  # plot x
                        clusters[idx][1, :],  # plot y
                        c=colors[k],
                        label=labels[k])

    # Plot centers
    ax.scatter(centers[0, :], centers[1, :], s=30, marker="x", label="centers")
    plt.xlabel("income")
    plt.ylabel("transactions")
    #plt.legend()
    plt.savefig(os.path.join(FIGURES_DIR, f"{str(UNIQUE_ID)}_{str(itr)}"))
    plt.clf()


def k_means_clustering(K, P_mxn, max_iter=MAX_ITERATIONS):
    M = P_mxn.shape[0] # cardinality
    N = P_mxn.shape[1] # dimensions

    # Initialize clusters and their centers
    clusters = {}
    C_nxk = np.array([]).reshape(N, 0)
    for k in range(1, K + 1):
        p = P_mxn[rand.randint(0, M-1)].reshape(-1, 1)
        clusters[k] = np.array([p]).reshape(N, -1)
        C_nxk = np.hstack((C_nxk, p))

    # Now, we iteratively update the clusters. We stop after
    # the cluster centers converge, or we reach a timeout.
    for itr in range(max_iter):
        plot_clusters(itr, clusters, C_nxk)
        # Compute Euclidean distances between all points and each center
        # i.e., each column is the distance of each point to kth center
        ED_mxk = np.array([]).reshape(M, 0)
        for k in range(K):
            # This computation makes me cringe. Why not use norm function?
            # This result is a vector of euclidean ED_mxk of M points
            d_point_to_center = np.sum((P_mxn - C_nxk[:, k])**2, axis=1)
            ED_mxk = np.c_[ED_mxk, d_point_to_center]

        # Generate cluster ids (1-based index)
        P_cluster_idx = np.argmin(ED_mxk, axis=1) + 1

        # Populate blobs (i.e., intermediate clusters)
        blobs = {}
        for k in range(1, K+1):
            blobs[k] = np.array([]).reshape(N, 0)

        for j, p in enumerate(P_mxn):
            cluster_idx = P_cluster_idx[j]
            blobs[cluster_idx] = np.hstack((blobs[cluster_idx], p.reshape(-1, 1)))

        # there's a weird step here, so ignoring it for now

        # Update centers based on blobs
        for k in range(1, K+1):
            if blobs[k].shape[1] != 0:
                C_nxk[:, k - 1] = np.mean(blobs[k], axis=1)

        clusters = blobs

    return clusters


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:\n\tkmeans.py <csv_data_filepath>")
        sys.exit(1)

    data = load_data(sys.argv[1])
    print(f"loaded data:\n{data.describe}")

    points = create_point_set(data, 3, 4)
    test_points = points[0:3]
    print(f"3 example points:\n{test_points}")

    k_clusters = k_means_clustering(K_CLUSTERS, points)

