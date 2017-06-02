import glob
import numpy as np
from PIL import Image
import tensorflow as tf
import math
import matplotlib.image as mpimg
import glob
import skimage.color as sk

def AUC(img1, img2):
    diff = np.square(img1-img2)
    auc = np.zeros(256)
    for i in range(256):
        auc[i] += diff[diff<=i/512].size
    return auc, np.sum(auc)/(img1.size*256)

def NN_ab(y,n=5):
    # y is [N, H, W, 3]
    NN_ab_x = np.round((y[:,:,:,1]+0.6)*19/1.2)
    NN_ab_y = np.round((y[:,:,:,2]+0.6)*19/1.2)
    NN_ab = NN_ab_x*19+NN_ab_y
    return NN_ab.astype(int)

def assign_prob(NN, y):
    # NN is [N, H, W]
    # y is [N, H, W, 3]
    prob_dist = np.zeros((y.shape[0]*y.shape[1]*y.shape[2], 400))
    NN = np.reshape(NN, [NN.size,])
    #NN =NN[...,np.newaxis]
    prob_dist[range(y.shape[0]*y.shape[1]*y.shape[2]),NN] = 1
    #above line doesnt work so to just test code did this
    #prob_dist[...,45] = 1
    return np.reshape(prob_dist,[y.shape[0], y.shape[1], y.shape[2], 400])

def Prob_dist(y):
    # Returns ab prob distribution for given training batch
    # y is [N, H, W,3] dim
    # x is [N, H, W, 400] dim
    NN = NN_ab(y)
    p = assign_prob(NN, y)
    return p
    
    