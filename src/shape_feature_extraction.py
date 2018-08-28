import numpy as np
from time import time
import scipy
from scipy.cluster.vq import vq, kmeans2
from .pcl_utils.build import pcl_utils as pcl

def extract_fphf_features(Vs, num_neighbors):
    pclUtils = pcl.PCLUtilsClass()
    pclUtils.estimate_fpfh_descriptors(Vs, num_neighbors)
    return pclUtils.get_fpfh_features()

### Extracts shape features from point set
def extract_descriptor_features(Vs, num_words, num_neighbors):


    descriptors = [extract_fphf_features(V, num_neighbors) for V in Vs]
    
    all_descriptors = np.concatenate(descriptors, axis=1)
    voc, label = scipy.cluster.vq.kmeans2(all_descriptors.transpose(), num_words)
    
    set_list = [V.shape[1] for V in Vs]
    splits = np.cumsum(np.array(set_list[:len(set_list)-1]))
    labels = np.split(label, splits)

    # Compute probabilities and store in sparse matrix
    feature_posteriors = [scipy.sparse.csr_matrix((np.ones(label.shape), (np.arange(len(label)), label)), shape=(len(label), num_words),dtype=np.float32) for label in labels]
    return feature_posteriors

def get_default_descriptor_distr(feature_likelihood, K):

    num_features = feature_likelihood[0].shape[1]
    sum_arr = np.array([f.sum(axis=0) for f in feature_likelihood]) #np.sum(num_features, axis=0)
    relative_distr = sum_arr.sum(axis=0)
    relative_distr = relative_distr / np.sum(relative_distr)
    
    desc_distr = np.random.gamma(np.tile(relative_distr.reshape(-1, 1), (1, K)), 1.0, size=(num_features,K))
    desc_distr_norm = np.sum(desc_distr, axis=0)
    desc_distr[:, desc_distr_norm == 0] = 1  # If all are zeros, set to uniform
    desc_distr = desc_distr / np.sum(desc_distr, axis=0)
    return desc_distr.astype('float32')
