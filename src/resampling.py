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
    farthest_pts = np.zeros((num_points_left, pts.shape[1]))
    farthest_pts[0] = pts[np.random.randint(len(pts))]
    distances = calc_distances(farthest_pts[0], pts)
    for i in range(1, num_points_left):
        farthest_pts[i] = pts[np.argmax(distances)]
        distances = np.minimum(distances, calc_distances(farthest_pts[i], pts))
    return farthest_pts
