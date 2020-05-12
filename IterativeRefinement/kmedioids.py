#!/usr/bin/python3
"""
src: https://towardsdatascience.com/k-medoids-clustering-on-iris-data-set-1931bf781e05

Clustering algorithm summary

Given a point set P, we want to create k-clusters such that
points belonging to each cluster have higher similarity than
those belonging to other clusters. Contrary to k-means
clustering, the median point of a cluster center defines
its center.

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
P_MINKOWSKI = 2
PLOT_FREQ = 5
FIGURES_DIR = "/home/bilkit/Workspace/PointClustering/IterativeRefinement/results"
SAVE = True
UNIQUE_ID = rand.randint(0, 100000)


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


def check_convergence(C, C_prime):
    # TODO: fix this stopping condition
    #return set([tuple(x) for x in C]) == set([tuple(x) for x in C_prime])
    return False


def compute_inner_cluster_distances(points, cluster_center):
    """
    Compute Euclidean distances between all points and each center
    # i.e., each column is the distance of each point to kth center
    Input
        mxn
            a set of points
        nx1
            a cluster center

    Output
        mx1 numpy array
        Column vector of distances between points and a cluster
        center.
    """
    # TODO: double check this assertion
    #assert(points.shape[0] == cluster_center.shape[0]), "N-dims (axis 0) are mis-matched"
    # This result is a vector (1xM) of euclidean ED_mxk of M points
    d_points_to_center = np.sum((points - cluster_center)**P_MINKOWSKI, axis=1)
    return d_points_to_center.reshape(-1, 1)


def compute_sse(clusters, centers):
    sse = 0 if len(clusters) > 0 else sys.float_info.max
    for k in clusters.keys():
        sse += np.sum(compute_inner_cluster_distances(clusters[k], centers[:, k - 1].reshape(-1, 1)))
    return sse


def k_map_clustering(K, P, key_points=None, max_iter=MAX_ITERATIONS):
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
        kxn numpy array
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
        D_mxk = np.array([]).reshape(M, 0)
        for k in range(K):
            D_k = compute_inner_cluster_distances(P, C_nxk[:, k])
            D_mxk = np.hstack((D_mxk, D_k))

        # Generate cluster ids (1-based index)
        P_cluster_idx = np.argmin(D_mxk, axis=1) + 1

        # Populate blobs (i.e., intermediate clusters)
        blobs = {}
        for k in range(1, K+1):
            blobs[k] = np.array([]).reshape(N, 0)

        for j, p in enumerate(P):
            cluster_idx = P_cluster_idx[j]
            blobs[cluster_idx] = np.hstack((blobs[cluster_idx], p.reshape(-1, 1)))

        # Update clusuter centers
        C_prev = C_nxk.copy()
        for k in range(1, K+1):
            if blobs[k].shape[1] == 0:
                continue

            # Search for new medioid amongst all cluster points
            k_idx = k - 1
            D_k_sum = np.sum(D_mxk[:, k_idx])
            for p in blobs[k].T:
                D_k_sum_prime = np.sum(compute_inner_cluster_distances(blobs[k], p.reshape(-1, 1)))

                if D_k_sum_prime < D_k_sum:
                    D_k_sum = D_k_sum_prime
                    C_nxk[:, k_idx] = p

        if check_convergence(C_prev, C_nxk):
            break
        else:
            clusters = blobs


        # Plot intermediate clusters
        if itr % PLOT_FREQ == 0:
            figure_filepath = os.path.join(FIGURES_DIR, f"kmed_{str(N)}d_{str(UNIQUE_ID)}_{str(itr)}")
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
                    return clusters, compute_sse(clusters, C_nxk)

    return clusters, compute_sse(clusters, C_nxk)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("""Usage:\n\tkmediods.py <int_mode>\n
modes: 
\t0 - marketing data (2D), 
\t1 - marketing data (3D), 
\t2 - bezier curves (2D), 
\t3 - bezier curves (3D), 
\t4 - bezier point cloud (3D)""")
        sys.exit(1)

    mode = int(sys.argv[1])
    if mode == 0:
        csv_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Mall_Customers.csv")
        data = load_data(csv_filepath)
        print(f"loaded data:\n{data.describe}")

        points_2d = create_point_set(data, 3, 4)
        k_clusters_2d, sse = k_map_clustering(K_CLUSTERS, points_2d)
    elif mode == 1:
        csv_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), )
        data = load_data(csv_filepath)
        print(f"loaded data:\n{data.describe}")

        points_3d = create_point_set(data, 3, 4, 2)
        k_clusters3d, sse = k_map_clustering(K_CLUSTERS, points_3d)
    elif mode == 2:
        curve = bezier_curve(10 * np.random.rand(5, 3))
        k_clusters3d, sse = k_map_clustering(K_CLUSTERS, curve)
    elif mode == 3:
        loop, control_points = bezier_looped_curve(n_dims=3)
        k_clusters3d, sse = k_map_clustering(K_CLUSTERS, loop, control_points)
    elif mode == 4:
        loop, control_points = bezier_looped_curve(n_dims=3)
        point_cloud = cloud_from_points(loop)
        k_clusters3d, sse = k_map_clustering(K_CLUSTERS, point_cloud, control_points)
    else:
        print(f"Unknown mode {mode}")
        sys.exit(1)

    print(f"sse: {sse: 0.4f}")
    print("Exit.")
