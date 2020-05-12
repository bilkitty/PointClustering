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
import Visualize.plotter_2d as plotter_2d
import Visualize.plotter_3d as plotter_3d

from matplotlib import pyplot as plt
from ToyRope.bezier_3d import bezier_curve, bezier_looped_curve
from ToyRope.utils import cloud_from_points

MAX_ITERATIONS = 40
K_CLUSTERS = 10
PLOT_FREQ = 5
FIGURES_DIR = "/home/bilkit/Workspace/PointClustering/Bfr/results"
SAVE = False
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


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("""Usage:\n\tkbfr.py <int_mode>\n
modes: 
\t0 - bezier curves (3D), 
\t1 - bezier point cloud (3D)""")
        sys.exit(1)

    mode = int(sys.argv[1])
    if mode == 0:
        csv_filepath = os.path.join(os.path.dirname(os.path.realpath(__file__)), )
        data = load_data(csv_filepath)
        print(f"loaded data:\n{data.describe}")

        points_2d = create_point_set(data, 3, 4)
        k_clusters_2d = k_means_clustering(K_CLUSTERS, points_2d)
    elif mode == 1:
        curve = bezier_curve(10 * np.random.rand(5, 3))
        k_clusters_3d = k_means_clustering(K_CLUSTERS, curve)
    elif mode == 2:
        loop, control_points = bezier_looped_curve(n_dims=3)
        k_clusters_3d = k_means_clustering(K_CLUSTERS, loop, control_points)
    elif mode == 3:
        loop, control_points = bezier_looped_curve(n_dims=3)
        point_cloud = cloud_from_control_points(loop)
        k_clusters_3d = k_means_clustering(K_CLUSTERS, loop, control_points)
    else:
        print(f"Unknown mode {mode}")
        sys.exit(1)

