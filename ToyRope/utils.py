import numpy as np


NORMAL_MU = 0
NORMAL_STD = 0.05


def cloud_from_points(points, std=NORMAL_STD):
    point_cloud = points
    for i in range(3):
        N = len(points)
        noise = np.column_stack((np.random.normal(NORMAL_MU, std, N),
                                np.random.normal(NORMAL_MU, std, N),
                                np.random.normal(NORMAL_MU, std, N)))
        point_cloud = np.vstack((point_cloud, points + noise))
    return point_cloud
