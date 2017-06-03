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
    #NN_ab_x = np.round((y[:,:,:,1]+0.6)*19/1.2)
    #NN_ab_y = np.round((y[:,:,:,2]+0.6)*19/1.2)
    #NN_ab = NN_ab_x*20+NN_ab_y
    NN_ab_x = np.round((y[:,:,:,1]+0.6)*27/1.2)
    NN_ab_y = np.round((y[:,:,:,2]+0.6)*27/1.2)
    NN_ab = NN_ab_x*28+NN_ab_y
    return NN_ab.astype(int)

def assign_prob(NN, y):
    # NN is [N, H, W]
    # y is [N, H, W, 3]
    prob_dist = np.zeros((y.shape[0]*y.shape[1]*y.shape[2], 784))
    NN = np.reshape(NN, [NN.size,])
    prob_dist[range(y.shape[0]*y.shape[1]*y.shape[2]),NN] = 1
    return np.reshape(prob_dist,[y.shape[0], y.shape[1], y.shape[2], 784])

def Prob_dist(y):
    # Returns ab prob distribution for given training batch
    # y is [N, H, W,3] dim
    # x is [N, H, W, 400] dim
    NN = NN_ab(y)
    p = assign_prob(NN, y)
    return p
    
def YUV2rgb(UV, y):
    # UV is (N, H, W, 400)
    # returns RGB values corresponding to this YUV input
    inv_mat  = np.array([[1,0,1.13983],[1,-0.39465,-0.58060],[1,2.03211,0]])
    UV_arg_max = np.argmax(UV,axis = 3)
    UV_arg_row_number = UV_arg_max//28
    UV_arg_column_number = UV_arg_max%28
    UV_val_a = -0.6 + UV_arg_row_number*1.2/27
    UV_val_b = -0.6 + UV_arg_column_number*1.2/27
    YUV_output = np.concatenate((y[...],UV_val_a[...,np.newaxis],UV_val_b[...,np.newaxis]),axis = 3)
    RGB_output = YUV_output.dot(inv_mat)
    RGB_output[RGB_output>1] = 1
    RGB_output[RGB_output<0] = 0
    print(np.amin(RGB_output))
    return RGB_output