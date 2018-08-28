from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from . import resampling as rs

def plotClouds(Vs, colors, fignum):
    color_list = ['r','g','b','c','y','m','k']    
    fig = plt.figure(fignum)
    ax = fig.add_subplot(111, projection='3d')
    if colors: 
        for V, color in zip(Vs, colors):
            ax.scatter(V[0,:], V[1,:], V[2,:], c=color, marker='o', s=1)
    else:
        for i, V in enumerate(Vs):
            cind = min(6,i)
            ax.scatter(V[0,:], V[1,:], V[2,:], c=color_list[cind], marker='o', s=1)


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def plotCloudsModel(Vs, X, fignum):
    color_list = ['r','g','b','c','y','m']    
    fig = plt.figure(fignum)
    ax = fig.add_subplot(111, projection='3d')

    for i, V in enumerate(Vs):
        Vv = rs.random_sampling(V, 2000) 
        cind = min(5,i)
        ax.scatter(Vv[0,:], Vv[1,:], Vv[2,:], c=color_list[cind], marker='o', s=4)

    ax.scatter(X[0,:], X[1,:], X[2,:], c='k', marker='o', s=16)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
