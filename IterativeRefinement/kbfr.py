#!/usr/bin/python3
"""
src: https://github.com/jeppeb91/bfr
     Refining Initial Points for K-Means Clustering
     Paul S. Bradley and Usama M. Fayyad (1998)


Clustering algorithm summary

Given a large point set P, we want to create k-clusters such that
points belonging to each cluster have higher similarity than those
belonging to other clusters.
NOTE: This algorithm assumes that clusters are normally distributed.

1. initialization of cluster params
2. assign points to clusters
    for each point
      compute distance to each cluster
      assign it to the nearest one
3. re-estimate cluster params
    for each cluster
      compute new centroid as avg of all cluster members
4. iterate 2-3 until timeout or model converges

"""
import numpy as np
import pandas as pd
import random as rand
import sys
import os
import bfr
import pickle
import Visualize.plotter_2d as plotter_2d
import Visualize.plotter_3d as plotter_3d

from ToyRope.bezier_3d import bezier_curve, bezier_looped_curve
from ToyRope.utils import cloud_from_points

K_CLUSTERS = 10
FIGURES_DIR = "/home/bilkit/Workspace/PointClustering/IterativeRefinement/results"
SAVE = True
UNIQUE_ID = rand.randint(0, 100000)
EPSILON = 0.002
BFR_INIT_ROUNDS = 10
BFR_MAHAL_FACTOR = 1.5
BFR_EUCL_THRESH = 0.2


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


def BFR(K, P, key_points=None, unique_id=UNIQUE_ID):
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
    pts = P.copy()
    plot_clusters = plotter_2d.plot_clusters if N == 2 else plotter_3d.plot_clusters

    bfr_model = bfr.Model(mahalanobis_factor=BFR_MAHAL_FACTOR,
                          euclidean_threshold=BFR_EUCL_THRESH,
                          merge_threshold=EPSILON,
                          dimensions=N,
                          init_rounds=BFR_INIT_ROUNDS,
                          nof_clusters=K)

    bfr_model.fit(P)

    # Finalize assigns clusters in the compress and retain set to the closest cluster in discard
    bfr_model.finalize()
    sse = bfr_model.error(P)

    cluster_indices = bfr_model.predict(P)
    centers = bfr_model.centers().reshape(N, -1)
    clusters = {}
    for k in range(1, K):
        clusters[k] = P[cluster_indices == k].reshape(N, -1)

    bfr_plot = bfr.plot.BfrPlot(bfr_model, P)
    bfr_plot.show()

    u_response = input("Save this figure? (Y/N/Q)")
    if u_response.lower() == 'y':
        # Save inputs to regenerate interactive figure later
        pickle_file = os.path.join(FIGURES_DIR, f"bfr_{str(N)}d_{str(unique_id)}.pickle")
        inputs = {
           "bfr_model" : bfr_model,
           "points" : P,
           "key_points" : key_points,
            "K" : K
        }
        pickle.dump(inputs, open(pickle_file, 'wb'))
        pickle_file.close()
        print(f"saved inputs to {pickle_file}")
    if u_response.lower() == 'q':
        print("Stopped clustering.")

    return clusters, sse


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("""Usage:\n\tkbfr.py <int_mode>\n
modes: 
\t0 - marketing data (2D),
\t1 - bezier curves (2D),
\t2 - bezier curves (3D), 
\t3 - bezier point cloud (3D)""")
        sys.exit(1)

    mode = int(sys.argv[1])
    if mode == 0:
        csv_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), "Mall_Customers.csv")
        data = load_data(csv_filepath)
        print(f"loaded data:\n{data.describe}")

        points_2d = create_point_set(data, 3, 4)
        k_clusters_2d, sse = BFR(K_CLUSTERS, points_2d)
    elif mode == 1:
        control_points = 10 * np.random.rand(5, 2)
        curve = bezier_curve(control_points)
        k_clusters_2d, sse = BFR(K_CLUSTERS, curve, control_points)
    elif mode == 2:
        loop, control_points = bezier_looped_curve(n_dims=3)
        k_clusters_3d, sse = BFR(K_CLUSTERS, loop, control_points)
    elif mode == 3:
        loop, control_points = bezier_looped_curve(n_dims=3)
        point_cloud = cloud_from_points(loop)
        k_clusters_3d, sse = BFR(K_CLUSTERS, loop, control_points)
    else:
        print(f"Unknown mode {mode}")
        sys.exit(1)

    print(f"sse: {sse: 0.4f}")
    print("Exit.")
