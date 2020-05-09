#!/usr/bin/python3

"""
Based on create spline tutorial
@ https://docs.pyvista.org/examples/00-load/create-spline.html

Required modules
python3 -m pip install pyvista, numpy, pypcd
"""

import pyvista as pv
import numpy as np
from pypcd import pypcd as pcd
import pprint

NORMAL_MU = 0
NORMAL_STD = 0.05
TH_RANGE = 1.5 * np.pi
Z_RANGE = 0.1
pp = pprint.PrettyPrinter(indent=2)


def make_points(N):
    """Helper to make XYZ points"""
    theta = np.linspace(-1 * TH_RANGE, TH_RANGE, N)
    z = np.linspace(-1 * Z_RANGE, Z_RANGE, N)
    r = z**2 + 1
    x = r * np.sin(theta)
    y = 1.5 * r * np.cos(theta)
    return np.column_stack((x, y, z))


def lines_from_points(points):
    """Given an array of points, make a line set"""
    poly = pv.PolyData()
    poly.points = points
    cells = np.full((len(points)-1, 3), 2, dtype=np.int)
    cells[:, 1] = np.arange(0, len(points)-1, dtype=np.int)
    cells[:, 2] = np.arange(1, len(points), dtype=np.int)
    poly.lines = cells
    return poly


def cloud_from_points(points):
    point_cloud = points
    for i in range(3):
        N = len(points)
        noise = np.column_stack((np.random.normal(NORMAL_MU, NORMAL_STD, N),
             np.random.normal(NORMAL_MU, NORMAL_STD, N),
             np.random.normal(NORMAL_MU, NORMAL_STD, N)))
        point_cloud = np.vstack((point_cloud, points + noise))
    return point_cloud


def display_spline(N):
    line = lines_from_points(make_points(N))
    line["scalars"] = np.arange(line.n_points)
    tube = line.tube(radius=0.1)
    tube.plot(smooth_shading=True)


def display_spline_points(N, filepath=""):
    point_cloud = cloud_from_points(make_points(N))

    if filepath is "":
        mesh = pv.PolyData(point_cloud)
        mesh.plot(eye_dome_lighting=True)
    else:
        uniform_color = np.repeat(1, len(point_cloud))[:, np.newaxis]
        rgb = np.hstack((uniform_color, uniform_color, uniform_color))
        encoded_colors = pcd.encode_rgb_for_pcl((rgb * 255).astype(np.uint8))
        colored_point_cloud = np.hstack((point_cloud, encoded_colors[:, np.newaxis])).astype(np.float32)
        pcd_point_cloud = pcd.make_xyz_rgb_point_cloud(colored_point_cloud)
        pp.pprint(f"Saving point cloud to '{filepath}'")
        pp.pprint(pcd_point_cloud.get_metadata())
        pcd_point_cloud.save(filepath)





