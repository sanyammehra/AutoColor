import tensorflow as tf
import numpy as np
import math
import matplotlib.image as mpimg
import glob
from data_utils import load_data, augment_data,normalise_train,normalise_test,rgb2gray
import matplotlib.pyplot as plt



''' a very baseline garbage model
'''
def baselinish(X,Y,is_training):
    
    W_conv = tf.get_variable("Wconv",shape = [3,3,1,32])
    b_conv = tf.get_variable("bconv",shape = [32])
    a1 = tf.nn.conv2d(X, W_conv, strides=[1,1,1,1], padding='SAME') + b_conv
    a1 = tf.nn.relu(a1)
    #a1 = tf.contrib.layers.batch_norm(a1,center = True, scale = True, is_training = is_training,scope = 'bn1')
    #a1 = tf.nn.max_pool(a1,ksize = [1,2,2,1],strides = [1,2,2,1],padding = 'VALID')
    
    W_conv2 = tf.get_variable("Wconv2",shape = [3,3,32,64])
    b_conv2 = tf.get_variable("bconv2",shape = [64])
    a1 = tf.nn.conv2d(a1, W_conv2, strides=[1,1,1,1], padding='SAME') + b_conv2
    a1 = tf.nn.relu(a1)
    #a1 = tf.contrib.layers.batch_norm(a1,center = True, scale = True, is_training = is_training,scope = 'bn2')
    
    W_conv3 = tf.get_variable("Wconv3",shape = [3,3,64,3])
    b_conv3 = tf.get_variable("bconv3",shape = [3])
    a1 = tf.nn.conv2d(a1, W_conv3, strides=[1,1,1,1], padding='SAME') + b_conv3
    
    return a1
    
    #a1 = tf.nn.relu(a1)
    #a1 = tf.contrib.layers.batch_norm(a1,center = True, scale = True, is_training = is_training,scope = 'bn3')

def lrelu(x, leak=0.2, name='lrelu'):
    return tf.maximum(leak*x, x)


def complex_pokemon_model(gray_image,train=True):
   conv1 = lrelu(slim.convolution(gray_image, 32, 3, stride=1, scope='conv1', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   conv2 = lrelu(slim.convolution(conv1, 32, 3, stride=1, scope='conv2', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   conv3 = lrelu(slim.convolution(conv2, 64, 3, stride=1, scope='conv3', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   conv4 = lrelu(slim.convolution(conv3, 64, 3, stride=1, scope='conv4', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   conv5 = lrelu(slim.convolution(conv4, 128, 3, stride=1, scope='conv5', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   conv6 = lrelu(slim.convolution(conv5, 128, 3, stride=1, scope='conv6', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   conv7 = lrelu(slim.convolution(conv6, 256, 3, stride=1, scope='conv7', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   conv8 = lrelu(slim.convolution(conv7, 256, 3, stride=1, scope='conv8', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   conv9 = lrelu(slim.convolution(conv8, 128, 3, stride=1, scope='conv9', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   conv10 = lrelu(slim.convolution(conv9, 128, 3, stride=1, scope='conv10', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   conv11 = lrelu(slim.convolution(conv10, 64, 1, stride=1, scope='conv11', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   conv12 = lrelu(slim.convolution(conv11, 64, 1, stride=1, scope='conv12', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   conv13 = lrelu(slim.convolution(conv12, 32, 1, stride=1, scope='conv13', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   conv14 = lrelu(slim.convolution(conv13, 32, 1, stride=1, scope='conv14', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   conv15 = lrelu(slim.convolution(conv14, 16, 1, stride=1, scope='conv15', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   conv16 = lrelu(slim.convolution(conv15, 16, 1, stride=1, scope='conv16', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   conv17 = lrelu(slim.convolution(conv16, 8, 1, stride=1, scope='conv17', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   if train: conv17 = tf.nn.dropout(conv17, 0.8)
   conv18 = lrelu(slim.convolution(conv17, 3, 1, stride=1, scope='conv18', normalizer_fn=slim.batch_norm, activation_fn=tf.identity))
   if train: conv18 = tf.nn.dropout(conv18, 0.8)
   
   tf.add_to_collection('vars', conv1)
   tf.add_to_collection('vars', conv2)
   tf.add_to_collection('vars', conv3)
   tf.add_to_collection('vars', conv4)
   tf.add_to_collection('vars', conv5)
   tf.add_to_collection('vars', conv6)
   tf.add_to_collection('vars', conv7)
   tf.add_to_collection('vars', conv8)
   tf.add_to_collection('vars', conv9)
   tf.add_to_collection('vars', conv10)
   tf.add_to_collection('vars', conv11)
   tf.add_to_collection('vars', conv12)
   tf.add_to_collection('vars', conv13)
   tf.add_to_collection('vars', conv14)
   tf.add_to_collection('vars', conv15)
   tf.add_to_collection('vars', conv16)
   tf.add_to_collection('vars', conv17)
   tf.add_to_collection('vars', conv18)
   
   return conv18
    