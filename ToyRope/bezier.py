#!/usr/bin/python3
import numpy as np
from scipy.special import comb
from matplotlib import pyplot as plt


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
    x_points = ctrl_points[:, 0]
    y_points = ctrl_points[:, 1]

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)])

    xvals = np.dot(x_points, polynomial_array)
    yvals = np.dot(y_points, polynomial_array)

    return np.vstack((xvals, yvals)).T


if __name__ == "__main__":
    nPoints = 4
    points = POINT_SCALE * np.random.rand(nPoints, 2)
    curve_points = bezier_curve(points, nTimes=1000)

    plt.plot(curve_points[:, 0], curve_points[:, 1])
    plt.plot(points[:, 0], points[:, 1], "ro")
    for nr in range(len(points)):
        plt.text(points[nr][0], points[nr][1], nr)

    plt.show()