#!/usr/bin/python3
"""
Evaluation of clustering algorithms:
For varied:
- point set size m
- cluster count k

Measures:
* sum of squared errors
* runtime
* histogram of cluster sizes
"""

import numpy as np
import random as rand
import sys

from matplotlib import pyplot as plt
from ToyRope.bezier_3d import bezier_unlooped_curve, bezier_looped_xy_plane
from IterativeRefinement.kmeans import k_means_clustering as kmeans
from IterativeRefinement.kmedoids import k_map_clustering as kmedoids
from IterativeRefinement.kbfr import BFR
from IterativeRefinement.linkage import linkage_clustering as LKG

MAX_ITERATIONS = 40
MAX_TRIALS = 15
K_UPPER_BOUND = 6
K_LOWER_BOUND = 3
RESOLUTION = 10000
UNIQUE_ID = rand.randint(0, 100000)


def run(k, loop, control_points, t_trials):
    # TODO: save these results as a matrix indexed
    # by an enum related to method name, sigh
    k_experiment = {
        "kmeans": np.zeros(t_trials),
        "kmedoids": np.zeros(t_trials),
        "bfr": np.zeros(t_trials),
        "lkg": np.zeros(t_trials)
    }

    for t in np.arange(t_trials):
        _, k_experiment["kmeans"][t] = kmeans(k, loop, control_points, MAX_ITERATIONS, UNIQUE_ID)
        _, k_experiment["kmedoids"][t] = kmedoids(k, loop, control_points, MAX_ITERATIONS, UNIQUE_ID)
        _, k_experiment["bfr"][t] = BFR(k, loop, control_points, UNIQUE_ID)
        _, k_experiment["lkg"][t] = LKG(k, loop, control_points, UNIQUE_ID)

        # TODO: runtimes

    return k_experiment


def experiment_on_simple(t_trials):
    loop, control_points = bezier_unlooped_curve(n_dims=3, t_resolution=RESOLUTION)

    k = 2**K_LOWER_BOUND
    experiment_results = {}
    while k <= 2**K_UPPER_BOUND:
        experiment_results[k] = run(k, loop, control_points, t_trials)
        k *= 2

    return experiment_results


def experiment_on_looped(t_trials):
    loop, control_points = bezier_looped_xy_plane(n_dims=3, t_resolution=RESOLUTION)

    k = 2**K_LOWER_BOUND
    experiment_results = {}
    while k <= (2**K_UPPER_BOUND):
        experiment_results[k] = run(k, loop, control_points, t_trials)
        k *= 2

    return experiment_results


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("""Usage:\n\tevaluate.py <int_mode>\n
modes: 
\t0 - simple case, 
\t1 - looped case""")
        sys.exit(1)

    mode = int(sys.argv[1])
    if mode == 0:
        results = experiment_on_simple(MAX_TRIALS)
    elif mode == 1:
        results = experiment_on_looped(MAX_TRIALS)
    else:
        print(f"Unknown mode {mode}")
        sys.exit(1)

    # Generate a plot of sse's vs k
    sse_kmeans = np.zeros((len(results), MAX_TRIALS))
    sse_kmedoids = np.zeros((len(results), MAX_TRIALS))
    sse_bfr = np.zeros((len(results), MAX_TRIALS))
    sse_lkg = np.zeros((len(results), MAX_TRIALS))
    i = 0
    for v in results.values():
        sse_kmeans[i][:] = v["kmeans"]
        sse_kmedoids[i][:] = v["kmedoids"]
        sse_bfr[i][:] = v["bfr"]
        sse_lkg[i][:] = v["lkg"]
        i += 1

    fig = plt.figure()
    ax = fig.add_subplot(111)
    kx = [2**i for i in range(K_LOWER_BOUND, K_UPPER_BOUND+1)]

    mean = np.mean(sse_kmeans, axis=1)
    std = np.std(sse_kmeans, axis=1)
    ax.fill_between(kx, mean - std, mean + std, alpha=0.5, label="kmeans")
    mean = np.mean(sse_kmedoids, axis=1)
    std = np.std(sse_kmedoids, axis=1)
    ax.fill_between(kx, mean - std, mean + std, alpha=0.5, label="kmedoids")
    mean = np.mean(sse_bfr, axis=1)
    std = np.std(sse_bfr, axis=1)
    ax.fill_between(kx, mean - std, mean + std, alpha=0.5, label="bfr")

    ax.set_xlabel('k clusters')
    ax.set_ylabel('sse')
    plt.xscale('symlog', basex=2)
    fig.suptitle("SSE w.r.t k-clusters")
    plt.legend()
    plt.savefig(f"/home/bilkit/Workspace/PointClustering/IterativeRefinement/eval_{mode}.png")
    plt.show()
    plt.close()

    print("Exit.")

