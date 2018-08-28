import numpy as np
from scipy.spatial import cKDTree

### Assuming uniform distribution (i.e JRMPC) ###
def default_uniform(points, args_tuple):
    return np.ones((len(points), 1))

### Empirical model ###
def empirical_estimate_kdtree(points, num_neighbors):
    tree = cKDTree(points)
    d, inds = tree.query(points, k=num_neighbors)

    A = np.stack([points[ind,:] for ind in inds])
    u, sigma, vh = np.linalg.svd(A - np.mean(A, axis=1, keepdims=True))
    sigma = np.sort(sigma, axis=1)
    point_weight = sigma[:,2] * sigma[:,1]
    point_weight_mat = np.stack([point_weight[ind] for ind in inds])

    return np.median(point_weight_mat, axis=1, keepdims=True)

### Sensor based models ###
def sensorLiDAR(points, args_tuple):
    from .pcl_utils.build import pcl_utils as pcl
    gamma, c = args_tuple
    points = points - c
    pclUtils = pcl.PCLUtils()
    pclUtils.compute_normals(points)
    normals = pclUtils.get_normals()
    X_norm_sqr = np.sum(points * points, 0)
    cos_ang_x = np.abs(np.sum(normals*points,0)) / np.sqrt(X_norm)
    sample_weights = X_norm_sqr / ((1-gamma) * cos_ang_x + gamma)

def sensorRGBD(points, args_tuple):
    from .pcl_utils.build import pcl_utils as pcl
    gamma, c = args_tuple
    points = points - c
    pclUtils = pcl.PCLUtils()
    pclUtils.compute_normals(points)
    normals = pclUtils.get_normals()
    X_norm = np.sqrt(np.sum(points * points, 0))
    cos_ang_x = abs(np.sum(normals*points,0))/X_norm
    sample_weights = X[3,:]*X[3,:]*X[3,:] / (X_norm*((1-gamma) * cos_ang_x + gamma))

