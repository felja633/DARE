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
            ax.scatter(V[0,:], V[1,:], V[2,:], c=color.transpose()/255.0, marker='o', s=9)
    else:
        for i, V in enumerate(Vs):
            cind = min(6,i)
            ax.scatter(V[0,:], V[1,:], V[2,:], c=color_list[cind], marker='o', s=9)


    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()

def plotCloudsModel(Vs, colors, X, fignum):
    color_list = ['r','g','b','c','y','m']    
    fig = plt.figure(fignum)
    ax = fig.add_subplot(111, projection='3d')

    if colors: 
        for V, color in zip(Vs, colors):
            Vc = rs.random_sampling([V,color], 2000)
            V = Vc[0]
            cc = Vc[1].transpose()
            ax.scatter(V[0,:], V[1,:], V[2,:], c=cc/255.0, marker='o', s=9)

    for i, V in enumerate(Vs):
        Vv = rs.random_sampling(V, 2000) 
        cind = min(5,i)
        ax.scatter(Vv[0,:], Vv[1,:], Vv[2,:], c=color_list[cind], marker='o', s=9)

    ax.scatter(X[0,:], X[1,:], X[2,:], c='k', marker='o', s=100)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
