#############################################################################################
# This demo script will run density adaptive point set registration of multiple point sets. # 
#############################################################################################

import numpy as np
from src import psreg
from src import point_cloud_plotting
from src import observation_weights
from src import resampling as rs
from src import data_loader as dl

# Main
if __name__ == "__main__":
    testLidar = True
    useFPS = False

    # Set this flag to use density adaptation
    useDARE = True
    # Set these flags to use color and/or geometrical features
    useColor = False 
    usefpfh = False  #not verified

    Vs = []
    Fs = []
    if testLidar:
        # Run demo on VPS outdoor datastet
        Vs, Fs = dl.load_demo_data("data/vps_out")
        Vs, Fs = zip(*[rs.random_sampling([V,F], 10000) for V,F in zip(Vs, Fs)]) 
        if useFPS:
            Vs, inds = zip(*[rs.farthest_point_sampling(V, 100) for V in Vs])
            Fs = [F[:,ind] for F, ind in zip(Fs,inds)]
    else:
        # Run demo on indoor tof datastet
        Vs, Fs = dl.load_demo_data("data/tofData")
        Vs, Fs = zip(*[rs.random_sampling([V,F], 10000) for V,F in zip(Vs, Fs)])    
    


    # Model parameters
    K = 300
    num_iters = 50
    gamma = float(0.005)
    epsilon = float(1e-5)
    num_channels = 4
    num_words = 10
    num_neighbors = 10
    
    # Init model
    pk = psreg.get_default_cluster_priors(K, gamma)
    X = psreg.get_randn_cluster_means(Vs, K)
    Q = psreg.get_default_cluster_precisions(Vs, X)
    Ps = psreg.get_default_start_poses(Vs, X)
    beta = psreg.get_default_beta(Q, gamma)

    # Features (no featurers results in JRMPC)
    fdistr = []
    features = []

    if useColor:
        # CPPSR 
        from src import color_feature_extraction as cfe
        print("initialize color features")
        color_features = cfe.channel_color_coding_hsv(Fs, num_channels)
        color_distr = cfe.get_default_color_distr(num_channels*num_channels*num_channels, K)
        features.append(color_features)        
        fdistr.append(color_distr)
        

    if usefpfh:
        # FPPSR not verified
        from src import shape_feature_extraction as sfe
        print("initialize descriptor features")
        descriptor_features = sfe.extract_descriptor_features(Vs, num_words, num_neighbors)
        descriptor_distr = sfe.get_default_descriptor_distr(descriptor_features, K)
        features.append(descriptor_features)
        fdistr.append(descriptor_distr)
        beta = beta / float(num_words)
    
    # Select observation weight function
    if useDARE:
        # DARE
        owf = observation_weights.empirical_estimate_kdtree
        num_neighbors = 10
    else:
        owf = observation_weights.default_uniform

    point_cloud_plotting.plotClouds(Vs, Fs, 52)
    # Create method object
    method = psreg.PSREG(beta, epsilon, pk, X, Q, fdistr,debug=False)
    # Register point sets
    print("register point sets...")
    TVs, X = method.register_points(Vs, features, num_iters, Ps, show_progress=True, observation_weight_function=owf, ow_args=num_neighbors)
    point_cloud_plotting.plotCloudsModel(TVs, Fs, X, 56)
