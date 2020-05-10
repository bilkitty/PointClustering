#!/usr/bin/python3
import numpy as np
from scipy.special import comb
from matplotlib import pyplot as plt
import plotter_3d

"""
Number of control points
The shape of Bezier curves are defined by N points
where N-1 points determine the highest order of the
polynomial.
e.g., N = 2 -> linear curve
      N = 3 -> quadratic curve
      N = 4 -> cubic curve
"""
N_CONTROL_POINTS = 6
N_DIMS = 3
POINT_SCALE = 200


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


if __name__ == "__main__":
    points = POINT_SCALE * np.random.rand(N_CONTROL_POINTS, N_DIMS)
    curve_points = bezier_curve(points, nTimes=1000)
    plotter_3d.plot_curve(curve_points, points)
