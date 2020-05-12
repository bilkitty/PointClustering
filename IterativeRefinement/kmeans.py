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
import Visualize.plotter_2d as plotter_2d
import Visualize.plotter_3d as plotter_3d

from matplotlib import pyplot as plt
from ToyRope.bezier_3d import bezier_curve, bezier_looped_curve
from ToyRope.utils import cloud_from_points

MAX_ITERATIONS = 40
K_CLUSTERS = 10
PLOT_FREQ = 5
FIGURES_DIR = "/home/bilkit/Workspace/PointClustering/KMeans/results"
SAVE = True
UNIQUE_ID = rand.randint(0, 100000)
CSV_FILENAME = "Mall_Customers.csv"


def load_data(filepath):
    assert(os.path.exists(filepath)), f"'{filepath}' does not exist"
    return pd.read_csv(filepath)


def create_point_set(csv_data, c1, c2, c3=-1):
    assert(0 <= c1 < csv_data.shape[1]), f"{c1} is out of bounds [0,{csv_data.shape[1]}]"
    assert(0 <= c2 < csv_data.shape[1]), f"{c2} is out of bounds [0,{csv_data.shape[1]}]"
    if c3 == -1:
        return csv_data.iloc[:, [c1, c2]].values
    else:
        assert(0 <= c3 < csv_data.shape[1]), f"{c3} is out of bounds [0,{csv_data.shape[1]}]"
        return csv_data.iloc[:, [c1, c2, c3]].values


def k_means_clustering(K, P, key_points, max_iter=MAX_ITERATIONS):
    """
    Computes K clusters from a point set P.
    Input
        int
            K number of cluster to compute
        mxn numpy array
            P points to cluster (m=cardinality, n=dims)
        int
            max_iter max iterations
    Output
        nxk dictionary
            K point clusters keyed by 1-based index
        kxn
            K centroid of each cluster (k=cluster count, n=dims)
    """
    M = P.shape[0] # cardinality
    N = P.shape[1] # dimensions
    plot_clusters = plotter_2d.plot_clusters if N == 2 else plotter_3d.plot_clusters

    # Initialize clusters and their centers
    clusters = {}
    C_nxk = np.array([]).reshape(N, 0)
    for k in range(1, K + 1):
        p = P[rand.randint(0, M-1)].reshape(-1, 1)
        clusters[k] = np.array([p]).reshape(N, -1)
        C_nxk = np.hstack((C_nxk, p))

    # Now, we iteratively update the clusters. We stop after
    # the cluster centers converge, or we reach a timeout.
    for itr in range(max_iter):
        # Compute Euclidean distances between all points and each center
        # i.e., each column is the distance of each point to kth center
        ED_mxk = np.array([]).reshape(M, 0)
        for k in range(K):
            # This computation makes me cringe. Why not use norm function?
            # This result is a vector of euclidean ED_mxk of M points
            d_point_to_center = np.sum((P - C_nxk[:, k])**2, axis=1)
            ED_mxk = np.c_[ED_mxk, d_point_to_center]

        # Generate cluster ids (1-based index)
        P_cluster_idx = np.argmin(ED_mxk, axis=1) + 1

        # Populate blobs (i.e., intermediate clusters)
        blobs = {}
        for k in range(1, K+1):
            blobs[k] = np.array([]).reshape(N, 0)

        for j, p in enumerate(P):
            cluster_idx = P_cluster_idx[j]
            blobs[cluster_idx] = np.hstack((blobs[cluster_idx], p.reshape(-1, 1)))

        # Update centers based on blobs
        C_prev = C_nxk
        for k in range(1, K+1):
            if blobs[k].shape[1] != 0:
                C_nxk[:, k - 1] = np.mean(blobs[k], axis=1)

        clusters = blobs


        # Plot intermediate clusters
        if itr % PLOT_FREQ == 0:
            figure_filepath = os.path.join(FIGURES_DIR, f"{str(N)}d_{str(UNIQUE_ID)}_{str(itr)}")
            if SAVE:
                plot_clusters(clusters, C_prev, figure_filepath, key_points)
            else:
                # Allow saving on the fly
                plot_clusters(clusters, C_prev, "", key_points)
                u_response = input("Save this figure? (Y/N/Q)")
                if u_response.lower() == 'y':
                    plot_clusters(clusters, C_prev, figure_filepath, key_points)
                if u_response.lower() == 'q':
                    print("Stopped clustering.")
                    return clusters

    return clusters


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("""Usage:\n\tkmeans.py <int_mode>\n
modes: 
\t0 - marketing data (2D), 
\t1 - marketing data (3D), 
\t2 - bezier curves (2D), 
\t3 - bezier curves (3D), 
\t4 - bezier point cloud (3D)""")
        sys.exit(1)

    mode = int(sys.argv[1])
    if mode == 0:
        csv_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), )
        data = load_data(csv_filepath)
        print(f"loaded data:\n{data.describe}")

        points_2d = create_point_set(data, 3, 4)
        k_clusters_2d = k_means_clustering(K_CLUSTERS, points_2d)
    elif mode == 1:
        csv_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), )
        data = load_data(csv_filepath)
        print(f"loaded data:\n{data.describe}")

        points_3d = create_point_set(data, 3, 4, 2)
        k_clusters_3d = k_means_clustering(K_CLUSTERS, points_3d)
    elif mode == 2:
        curve = bezier_curve(10 * np.random.rand(5, 3))
        k_clusters_3d = k_means_clustering(K_CLUSTERS, curve)
    elif mode == 3:
        loop, control_points = bezier_looped_curve(n_dims=3)
        k_clusters_3d = k_means_clustering(K_CLUSTERS, loop, control_points)
    elif mode == 4:
        loop, control_points = bezier_looped_curve(n_dims=3)
        point_cloud = cloud_from_points(loop)
        k_clusters_3d = k_means_clustering(K_CLUSTERS, point_cloud, control_points)
    else:
        print(f"Unknown mode {mode}")
        sys.exit(1)

    print("Exit.")
