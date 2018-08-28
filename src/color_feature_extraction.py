import colorsys
import numpy as np
from time import time

class ChannelEncoder:
    def __init__(self,
                 nchans=None,
                 bounds=None,
                 mflag=None,
                 cscale=1.0):
        """

        """
        self.mflag = mflag
        self.nchans = nchans
        self.bounds = bounds
        self.cscale = cscale
        self.bfuncwidth = 1.5
        self.d = self.cscale * self.bfuncwidth

        if mflag == 0:
            self.fpos = (self.bounds[1] + self.bounds[0] * self.nchans - self.d * (self.bounds[0] + self.bounds[1])) / (self.nchans + 1 - 2 * self.d)
            self.ssc = (self.bounds[1] - self.bounds[0]) / (self.nchans + 1 - 2 * self.d)
        else:
            self.ssc = (self.bounds[1] - self.bounds[0]) / self.nchans
            self.fpos = self.bounds[0]

    def basis_cos2(self, x):

        c = np.cos(np.pi / 3 * x)
        val = c * c * (np.abs(x) < self.bfuncwidth)
        return val

    def basis_bs2(self, x):
        y = (np.abs(x) < 1.0/2.0) * (3.0/4.0 - np.abs(x)**2) + (np.abs(x) >= 1.0/2.0) * (np.abs(x) <= 3.0/2.0) * ((3.0/2.0 - abs(x))**2)/2.0
        return y

    ### Encode a value to a channel representation
    def encode(self, x):

        #cc = np.zeros((len(x), self.nchans))
        val = (x - self.fpos) / self.ssc + 1
        cpos  = np.arange(self.nchans) + 1
        
        cpos = cpos.reshape(1, self.nchans)
        val = val.reshape(len(val),1)
        
      
        if self.mflag:
            ndist = self.nchans / 2.0 - np.abs(np.mod(cpos - val, self.nchans) - self.nchans / 2.0)
        else:
            ndist = np.abs(cpos - val)

        
        return self.basis_bs2(ndist)


def generate_1d_channels(feature_map, nch, max_v, min_v, modulo):
    
    not_mod = (1-modulo)
    num_ext_channels = nch + 2*not_mod
    che = ChannelEncoder(num_ext_channels, [min_v, max_v], modulo)
    return che.encode(feature_map) 

def uniform_channel_coding(feature_map, num_channels, modulo):

    ### Do this per point...
    cc1 = generate_1d_channels(feature_map[0,:], num_channels, 1.0, 0.0, modulo[0])
    cc2 = generate_1d_channels(feature_map[1,:], num_channels, 1.0, 0.0, modulo[1])
    cc3 = generate_1d_channels(feature_map[2,:], num_channels, 1.0, 0.0, modulo[2])

    nmodulo = [1 - m for m in modulo]
    nch1 = num_channels + 2 * nmodulo[0]
    nch2 = num_channels + 2 * nmodulo[1]
    nch3 = num_channels + 2 * nmodulo[2]
    nch = [nch1,nch2,nch3]
    num_points = len(cc1)
    ### compute outer products of channels
    cc1cc2kron = cc2.reshape((len(cc2),nch2, 1)) * cc1.reshape((num_points, 1, nch1))
    tmp = cc1cc2kron.reshape((num_points, 1, nch2, nch1))
    channels = cc3.reshape((num_points, nch3, 1, 1)) * tmp

    weights = np.ones((channels.shape[0],num_channels,num_channels,num_channels)) * num_channels * 6.0/5.0
    weights[:,nmodulo[2]:weights.shape[1]-nmodulo[2], nmodulo[1]:weights.shape[2]-nmodulo[1], nmodulo[0]:weights.shape[3]-nmodulo[0]] = num_channels 
    channels = channels[:, nmodulo[2]:channels.shape[1]-nmodulo[2], nmodulo[1]:channels.shape[2]-nmodulo[1], nmodulo[0]:channels.shape[3]-nmodulo[0]]

    channels = channels * weights * 19.200233330189796

    flatt_channels = channels.reshape((channels.shape[0], num_channels**3))
    
    return flatt_channels

def channel_color_coding_rgb(feature_data, num_channels):

    modulo = [0, 0, 0]
    channel_map = np.ndarray(len(feature_data), dtype=object)

    for i, feature_map in enumerate(feature_data):
        feature_map = feature_map/255.0
        
        channel_map[i] = uniform_channel_coding(feature_map, num_channels, modulo)

    return channel_map

def channel_color_coding_hsv(feature_data, num_channels):


    modulo = [1, 0, 0]
    channel_map = np.ndarray(len(feature_data), dtype=object)

    for i, feature_map in enumerate(feature_data):
        feature_map = feature_map/255.0
        
        feature_map = [colorsys.rgb_to_hsv(r, g, b) for (r, g, b) in feature_map.transpose()]
        channel_map[i] = uniform_channel_coding(np.array(feature_map).transpose(), num_channels, modulo).astype('float32')

    return channel_map

def get_gamma_color_distr(num_features, K):

    color_distr = np.random.gamma(1.0,1.0,size=(num_features,K))
    color_distr_norm = np.sum(color_distr, axis=0)
    color_distr[:, color_distr_norm == 0] = 1.0  # If all are zeros, set to uniform
    color_distr = color_distr / np.sum(color_distr, axis=0)
    return color_distr.astype('float32')

def get_default_color_distr(num_features, K):
    color_distr = 1.0/num_features * np.ones((num_features, K))
    return color_distr.astype('float32')
