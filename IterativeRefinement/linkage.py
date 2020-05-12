#!/usr/bin/python3
"""
src: https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html 
     https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/

Clustering algorithm summary

TODO
"""
import numpy as np
import random as rand
import sys
import os
import Visualize.plotter_2d as plotter_2d
import Visualize.plotter_3d as plotter_3d

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from ToyRope.bezier_3d import bezier_curve, bezier_looped_curve
from ToyRope.utils import cloud_from_points

MAX_ITERATIONS = 40
K_CLUSTERS = 10
P_MINKOWSKI = 2
PLOT_FREQ = 5
FIGURES_DIR = "/home/bilkit/Workspace/PointClustering/IterativeRefinement/results"
SAVE = True
UNIQUE_ID = rand.randint(0, 100000)


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


def single_linkage_clustering(K, P, key_points=None):
    M = P.shape[0] # cardinality
    N = P.shape[1] # dimensions
    plot_clusters = plotter_2d.plot_clusters if N == 2 else plotter_3d.plot_clusters

    Z = linkage(P, 'single')
    #dn = dendrogram(Z)
    cluster_indices = fcluster(Z, K, criterion='maxclust')

    clusters = {}
    for k in range(1, K):
        clusters[k] = P[cluster_indices == k].reshape(N, -1)

    figure_filepath = os.path.join(FIGURES_DIR, f"lkg_{str(N)}d_{str(UNIQUE_ID)}") if SAVE else ""
    plot_clusters(clusters, np.zeros(3).reshape(3,1), figure_filepath, key_points)

    # TODO: compute sse
    return clusters, 0

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("""Usage:\n\tlinkage.py <int_mode>\n
modes: 
\t0 - bezier curves (2D), 
\t1 - bezier curves (3D), 
\t2 - bezier point cloud (3D)""")
        sys.exit(1)

    mode = int(sys.argv[1])
    if mode == 0:
        curve = bezier_curve(10 * np.random.rand(5, 3))
        k_clusters3d, sse = single_linkage_clustering(K_CLUSTERS, curve)
    elif mode == 1:
        loop, control_points = bezier_looped_curve(n_dims=3)
        k_clusters3d, sse = single_linkage_clustering(K_CLUSTERS, loop, control_points)
    elif mode == 2:
        loop, control_points = bezier_looped_curve(n_dims=3)
        point_cloud = cloud_from_points(loop)
        k_clusters3d, sse = single_linkage_clustering(K_CLUSTERS, point_cloud, control_points)
    else:
        print(f"Unknown mode {mode}")
        sys.exit(1)

    print(f"sse: {sse: 0.4f}")
    print("Exit.")
