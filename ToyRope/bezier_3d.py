#!/usr/bin/python3
import numpy as np
from scipy.special import comb
from matplotlib import pyplot as plt
import Visualize.plotter_3d as plotter_3d
from ToyRope.utils import cloud_from_points
import sys

"""
Number of control points
The shape of Bezier curves are defined by N points
where N-1 points determine the highest order of the
polynomial.
e.g., N = 2 -> linear curve
      N = 3 -> quadratic curve
      N = 4 -> cubic curve
"""
N_CONTROL_POINTS = 4
N_DIMS = 3
POINT_SCALE = 20
LOOP_MIN_SEP = 0.3
LOOP_MAX_SEP = 0.8
DEMO_LOOP = True


def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


def bezier_curve(ctrl_points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(ctrl_points)
    n_dims = len(ctrl_points[0])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])

    curve = np.array([]).reshape(nTimes, 0)
    for d in range(ctrl_points.shape[1]):
        dim = np.dot(ctrl_points[:, d], polynomial_array).reshape(-1, 1)
        curve = np.hstack((curve, dim))

    return curve

def bezier_looped_curve(n_dims, t_resolution=1000):
    c0 = POINT_SCALE * np.random.rand(1, n_dims)
    c4 = POINT_SCALE * np.random.rand(1, n_dims)
    rand_axis = np.random.randint(0, n_dims)
    rand_shift = POINT_SCALE * np.random.uniform(0, 1)
    next_shift = np.subtract(c0, c4)
    rand_offset = np.random.uniform(LOOP_MIN_SEP, LOOP_MAX_SEP) * (next_shift - np.ones(n_dims))
    return bezier_looped_curve_fixed(c0, c4, n_dims, rand_shift, rand_axis, rand_offset, t_resolution)

def bezier_looped_xy_plane(n_dims, t_resolution=1000):
    c0 = np.zeros(3)
    c4 = np.zeros(3)
    z_offset = np.zeros(3)
    c0[0] = 0.25; c0[1] = 0.25
    c4[0] = 0.25; c4[1] = 0.75
    z_offset[1] = 2;z_offset[2] = 0 # leaving this at zero for now for the purspose of illustration
    c0 = POINT_SCALE * c0
    c4 = POINT_SCALE * c4
    x_axis = 0
    shift_by = 0.75
    next_shift = np.subtract(c0, c4)
    return bezier_looped_curve_fixed(c0, c4, n_dims, shift_by, x_axis, z_offset, t_resolution)

def bezier_unlooped_curve(n_dims, t_resolution=1000):
    c0 = np.zeros(3)
    c4 = np.zeros(3)
    c0[0] = 0.25; c0[1] = 0.25
    c4[0] = 0.25; c4[1] = 0.75
    c0 = POINT_SCALE * c0
    c4 = POINT_SCALE * c4
    x_axis = 0
    shift_by = 0.75
    next_shift = np.subtract(c0, c4)
    z_offset = 0.9 * (next_shift + np.ones(n_dims))
    return bezier_looped_curve_fixed(c0, c4, n_dims, shift_by, x_axis, z_offset, t_resolution)

def bezier_looped_curve_fixed(c0, c4, n_dims, shift, axis, loop_offset, t_resolution=1000):
    """
    Generate a Bezier curve with a loop. Note that order of
    control points matters here.
    We achieve a loop by:
    1) create start and end point
    2) add two points to form a box in a plane
    3) perform transform to twist the box
    4) add fifth point to get 3d loop
    """
    axis_shift = np.zeros((2, n_dims))
    axis_shift[:, axis] = shift

    next_shift = np.subtract(c0, c4)
    c1 = c0 - next_shift
    c2 = c4 + next_shift
    c3 = c0 + loop_offset
    c1 = c1 + axis_shift[0]
    c2 = c2 + axis_shift[1]

    ctrl_points = np.vstack((c0, c1))
    ctrl_points = np.vstack((ctrl_points, c2))
    ctrl_points = np.vstack((ctrl_points, c3))
    ctrl_points = np.vstack((ctrl_points, c4))

    m_points = len(ctrl_points)

    t = np.linspace(0.0, 1.0, t_resolution)

    polynomial_array = np.array([bernstein_poly(i, m_points-1, t) for i in range(0, m_points)])

    curve = np.array([]).reshape(t_resolution, 0)
    for d in range(ctrl_points.shape[1]):
        dim = np.dot(ctrl_points[:, d], polynomial_array).reshape(-1, 1)
        curve = np.hstack((curve, dim))

    return curve, ctrl_points


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("""Usage:\n\tbezier_3d.py <int_mode>\n
    modes: 
    \t0 - simple curve, 
    \t1 - looped curve, 
    \t2 - point cloud,
    \t3 - looped point cloud, 
    \t4 - simple curve (fixed), 
    \t5 - looped curve (fixed)""")
        sys.exit(1)

    mode = int(sys.argv[1])
    if mode == 0:
        points = POINT_SCALE * np.random.rand(N_CONTROL_POINTS, N_DIMS)
        curve_points = bezier_curve(points, nTimes=1000)
        plotter_3d.plot_curve(curve_points, points)
    elif mode == 1:
        loop, points = bezier_looped_curve(n_dims=3)
        plotter_3d.plot_curve(loop, points)
    elif mode == 2:
        points = POINT_SCALE * np.random.rand(N_CONTROL_POINTS, N_DIMS)
        curve_points = bezier_curve(points, nTimes=1000)
        point_cloud = cloud_from_points(curve_points)
        plotter_3d.plot_curve(point_cloud, points)
    elif mode == 3:
        loop, points = bezier_looped_curve(n_dims=3)
        point_cloud = cloud_from_points(loop)
        plotter_3d.plot_curve(point_cloud, points)
    elif mode == 4:
        loop, points = bezier_unlooped_curve(n_dims=3)
        plotter_3d.plot_curve(loop, points)
    elif mode == 5:
        loop, points = bezier_looped_xy_plane(n_dims=3)
        plotter_3d.plot_curve(loop, points)
    else:
        print(f"unknown mode {mode}")

