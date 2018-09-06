## resampling ##
import numpy as np

def random_sampling(pts, num_points_left):
    
    def rnd_samp(nump, s):

        arr = np.arange(0, nump)
        if nump < s:
            return arr

        arr = np.random.permutation(arr)
        return arr[0:s]

    if isinstance(pts, (list,)):
        sh = pts[0].shape
        arr = rnd_samp(sh[1], num_points_left)
        return [p[:,arr] for p in pts]
    else:
        sh = pts.shape
        arr = rnd_samp(sh[1], num_points_left)
        return pts[:,arr]


def farthest_point_sampling(pts, num_points_left):
    """
    Select a subset of K points from pts using farthest point sampling
    from: https://codereview.stackexchange.com/questions/179561/farthest-point-algorithm-in-python
    """
    def calc_distances(p0, points):
        return ((p0[:, np.newaxis] - points)**2).sum(axis=0)

    farthest_pts = np.zeros((pts.shape[0], num_points_left))
    f_ind = np.random.randint(pts.shape[1])
    farthest_pts[:,0] = pts[:,f_ind]
    distances = calc_distances(farthest_pts[:,0], pts)
    inds = []
    inds.append(f_ind)
    for i in range(1, num_points_left):
        farthest_pts[:,i] = pts[:,np.argmax(distances)]
        inds.append(np.argmax(distances))
        distances = np.minimum(distances, calc_distances(farthest_pts[:,i], pts))

    return farthest_pts, inds
