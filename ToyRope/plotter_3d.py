import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_curve(bezier_curve, ctrl_points):
    assert(bezier_curve.shape[1] == 3), f"expected 3-d bezier, got: {bezier_curve.shape[1]}"
    assert(ctrl_points.shape[1] == 3), f"expected 3-d points, got: {ctrl_points.shape[1]}"

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot Bezier curve
    X = bezier_curve[:, 0]
    Y = bezier_curve[:, 1]
    Z = bezier_curve[:, 2]
    ax.plot(X, Y, Z)
    #from matplotlib import cm
    #cset = ax.contour(X, Y, Z, zdir='z', offset=np.min(Z), cmap=cm.coolwarm)
    #cset = ax.contour(X, Y, Z, zdir='x', offset=np.min(X), cmap=cm.coolwarm)
    #cset = ax.contour(X, Y, Z, zdir='y', offset=np.min(Y), cmap=cm.coolwarm)

    # Plot control points
    ax.scatter(ctrl_points[:, 0], ctrl_points[:, 1], ctrl_points[:, 2], c="red")
    for nr in range(len(ctrl_points)):
        ax.text(ctrl_points[nr][0], ctrl_points[nr][1], ctrl_points[nr][2], s=str(nr))

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    plt.show()


