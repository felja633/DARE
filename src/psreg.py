##############################################
#  Density Adaptative Point Set Registration #
##############################################
import sys
import numpy as np
from numpy.linalg import svd, det
from time import time
from . import observation_weights
from . import point_cloud_plotting
from scipy.spatial import cKDTree
from .math_utils.build import math_utils

def list_prod(X):
    if len(X)==1:
        return X[0]
    elif len(X) == 0:
        return 1.0
    else:
        return np.prod(np.stack(X, 2), 2)

def sqe(Y, X):
    d = Y[:, :, None].transpose(1, 2, 0) - X[:, :, None].transpose(2, 1, 0)
    s = np.sum(d * d, axis=2)
    
    return s

def sqe2(Y, X):
    return math_utils.mat_sqe(Y.transpose(), X.transpose())

def get_default_cluster_priors(num_clusters, gamma):
    pk = 1 / (num_clusters + gamma) * np.ones((num_clusters, 1), dtype=np.float32)
    return pk.transpose()


def get_randn_cluster_means(point_clouds, num_clusters):
    """ Create random cluster means, distributed on a sphere.
        The standard deviation of all point-cloud points is the sphere radius.
    :param point_clouds: [ X1, X2, ... ]. Xi = 3 x Ni points [np.array].
    :param num_clusters: Number of clusters to generate
    :return: cluster means, (3, num_clusters) [np.array]
    """
    # Sample the the unit sphere and scale with data standard deviation
    X = np.random.randn(3, num_clusters).astype(np.float32)
    X = X / np.linalg.norm(X, axis=0)
    v = np.var(np.concatenate(point_clouds, 1), 1, keepdims=True)
    means = X * np.sqrt(v)
    return means


def get_default_cluster_precisions(point_clouds, cluster_means):

    # Minimum coordinates in point clouds and clusters
    min_xyz = [np.min(pcl, 1) for pcl in point_clouds]  # list of per-pcl minima
    min_xyz = min_xyz + [np.min(cluster_means, 1)]  # append cluster_means minima
    min_xyz = np.min(np.stack(min_xyz), 1)  # get joint minimum

    # Maximum coordinates in point clouds and clusters
    max_xyz = [np.max(pcl, 1) for pcl in point_clouds]
    max_xyz = max_xyz + [np.max(cluster_means, 1)]
    max_xyz = np.max(np.stack(max_xyz), 1)

    q = 1 / sqe(min_xyz[...,np.newaxis], max_xyz[...,np.newaxis])

    Q = q * np.ones((cluster_means.shape[1], 1))
    return Q.astype(np.float32)


def get_default_start_poses(point_clouds, cluster_means):
    """ Create default start poses
    :param cluster_means:
    :param point_clouds:
    :return:
    """
    I = np.eye(3, dtype=np.float32)  # Identity rotation
    mu = np.mean(cluster_means, 1)  # Mean of cluster means
    poses = [(I, mu - np.mean(pcl, 1)) for pcl in point_clouds]
    return poses


def get_default_beta(cluster_precisions, gamma):

    h = 2 / np.mean(cluster_precisions)
    beta = gamma / (h * (gamma + 1))
    return float(beta)

class PSREG:

    def __init__(self,
                 betas=None,
                 epsilon=None,
                 cluster_priors=None,
                 cluster_means=None,
                 cluster_precisions=None,
                 feature_distr=None,
                 debug=False,
                 use_kdtree=False,
                 kd_rate=0.1,
                 fix_cluster_pos_iter=2):
        """
        :param beta:
        :param epsilon:
        :param cluster_priors:      (1,K) numpy.array (\rho_k)
        :param cluster_means:       (3,K) numpy.array (X)
        :param cluster_precisions:  (3,K) numpy.array (Q)
        """

        self.betas = betas
        self.epsilon = epsilon
        self.cluster_priors = cluster_priors
        self.cluster_means = cluster_means
        self.cluster_precisions = cluster_precisions
        self.feature_distr = feature_distr
        self.debug = debug
        self.kd_rate = kd_rate
        self.use_kdtree = use_kdtree
        self.fix_cluster_pos_iter = fix_cluster_pos_iter

    def register_points(self, point_clouds, feature_likelihoods, num_iters, start_poses, show_progress=False, observation_weight_function=observation_weights.default_uniform, ow_args=()):
        """
        :param point_clouds: [ X1, X2, ... ]. Xi = (3, Ni) numpy.array
        :param num_iters:  Number of iterations to run
        :param start_poses: [ (R1, t1), (R2, t2) ... ]
          Ri = pcl-to-world rotation (3,3)  numpy.array,
          ti = pcl-to-world translation vector (3,1) numpy.array
        :return:
        """
        N = len(point_clouds)

        Vs = point_clouds
        Ps = start_poses

        pk = self.cluster_priors
        X = self.cluster_means
        Q = self.cluster_precisions
        fd = self.feature_distr
        ow_reg_factor = 8.0

        fts = feature_likelihoods

        # Compute the observation weights
        observation_weights = [observation_weight_function(V.transpose(), ow_args) for V in Vs]

        TVs = [R @ V + t[..., np.newaxis] for V, (R, t) in zip(Vs, Ps)]
        
        for ow in observation_weights:
            m = np.sum(ow)/ow.shape[0]
            ow[np.where(ow > m * ow_reg_factor)] = m * ow_reg_factor

        ds = [sqe2(TV, X) for TV in TVs]
        t_tot = time()
        for i in range(num_iters):
            t0 = time()
            
            a_s, Ls, Rs, ts, TVs, X, Q, den, fd, ds = self._iterate(TVs, X, pk, Q, fd, Vs, fts, ds, observation_weights, i)
            if show_progress:
                print("%03d: %.1f ms" % (i+1, (time() - t0) * 1000))

        tot_time = time() - t_tot 
        print("tot time %03d: %.1f ms" % (i+1, (tot_time) * 1000))

        if self.debug:
            point_cloud_plotting.plotCloudsModel(TVs, X, 56)
        return TVs, X

    # (uniform priors so far...)
    def _iterate(self, TVs, X, pk, Q, feature_distr, Vs, features, ds, ows, current_iter):
        """ Run one cppsr iteraton """

        M = len(TVs)

        a_s = np.ndarray(M, dtype=object)
        Ls = np.ndarray(M, dtype=object)
        Rs = np.ndarray(M, dtype=object)
        ts = np.ndarray(M, dtype=object)
        TV2s = np.ndarray(M, dtype=object)
        ac_den = np.ndarray(M, dtype=object)
        ap = np.ndarray(M, dtype=object)
        num_features = len(feature_distr)
        pyz_feature = np.ndarray((M, num_features), dtype=object)
        
        Qt = Q.transpose()
        for i, (TV, V, d, ow) in enumerate(zip(TVs, Vs, ds, ows)):

            # Posteriors
            a = pk * np.power(Qt, 1.5) * math_utils.mat_exp(-0.5 * Qt * d)

            ap[i] = a.copy()

            if features:
                for j, (fl, fd) in enumerate(zip(features, feature_distr)):
                    # the joint feature distribution p(y|z,th)
                    pyz_feature[i][j] = fl[i] @ fd 

                a = list_prod(pyz_feature[i]) * a

            
            ac_den[i] = np.sum(a, 1, keepdims=True) + self.betas
            a = a / ac_den[i]  # normalize row-wise
            a = a * ow # apply observation weights
  
            L = np.sum(a, 0, keepdims=True).transpose()
            W = (V @ a) * Qt

            b = L * Q  # weights, b
            mW = np.sum(W, 1, keepdims=True)  # mean of W
            mX = X @ b  # mean of X
            z = L.transpose() @ Q  # sumOfWeights
            P = (X @ W.transpose()) - (mX @ mW.transpose()) / z

            # Compute R and t
            uu, _, vv = svd(P)
            
            vv = vv.transpose()  # Note: v is transposed compared to matlab's svd()
            S = np.diag([1, 1, det(uu @ vv)]).astype('float32')
            R = uu @ S @ vv.transpose()
            R = R
            t = (mX - R @ mW) / z
            TV = R @ V + t  # transform V

            a_s[i] = a
            Ls[i] = L
            Rs[i] = R
            ts[i] = t
            TV2s[i] = TV

        TVs = TV2s

        # Update X

        den = Ls[0].copy()
        for L in Ls[1:]:
            den += L
        den = den.transpose()
        
        if self.fix_cluster_pos_iter < current_iter:
            X = TVs[0] @ a_s[0]
            for TV, a in zip(TVs[1:], a_s[1:]):
                X += TV @ a
            X = X / den
        # Update Q
        
        ds2 = [sqe2(TV, X) for TV in TVs]
        wn = np.sum(a_s[0] * ds2[0], 0, keepdims=True)

        for distances, a in zip(ds2[1:], a_s[1:]):
            wn += np.sum(a * distances, 0, keepdims=True)

        Q = (3 * den / (wn + 3 * den * self.epsilon)).transpose()

        if features:
            for j, fd in enumerate(feature_distr):
                ac_sum = np.zeros(fd.shape)
                indlist = np.arange(0, num_features)
                
                # Update feature distributions
                for i, (TV, V, ow) in enumerate(zip(TVs, Vs, ows)):
                    normed = features[j][i] / ac_den[i]
                    ac_sum = ac_sum + normed.transpose(1, 0) @ (ow * ap[i] * list_prod(pyz_feature[i][indlist != j]))

                fd = np.multiply(fd, ac_sum)
                fd = fd / (np.sum(fd, axis=0) + 0.0000001)
            
        return a_s, Ls, Rs, ts, TVs, X, Q, den, feature_distr, ds2

