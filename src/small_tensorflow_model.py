import tensorflow as tf
import numpy as np
import math
import matplotlib.image as mpimg
import glob




def read_data():
	input_image = []
	i = 0
	for filename in glob.iglob('../data/tiny-imagenet-200/train/**/*.JPEG',recursive=True):
		img = mpimg.imread(filename)
		if(img.shape == (64,64,3)):
			input_image.append(list(mpimg.imread(filename)))
		if(i%1000 == 0):
			print(i)
		i+=1
		if(i>5000):
			break

	return np.asarray(input_image)

image_data = read_data()
print (image_data.shape)





def simple_rgb_to_gray(X,Y):
	Wconv1 = tf.get_variable("Wconv1", shape=[1, 1, 3, 1])
	bconv1 = tf.get_variable("bconv1", shape=[1])
	y_out = tf.nn.conv2d(X, Wconv1, strides=[1,1,1,1], padding='SAME')
	y_out = y_out[:,:,:,0]
	return y_out


X = tf.placeholder(tf.float32, [None, 64, 64, 3])
Y = tf.placeholder(tf.float32 ,[None , 64,64])
y_out = simple_rgb_to_gray(X,Y)
loss = tf.nn.l2_loss(y_out - Y)
optimiser = tf.train.AdamOptimizer(0.2)
train_step = optimiser.minimize(loss)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])


sess = tf.Session()
sess.run(tf.global_variables_initializer())


for i in range(200):

	feed_dict = {X: image_data[0:100], Y: rgb2gray(image_data[0:100])}
	losses,_ = sess.run([loss,train_step],feed_dict)
	if(i%50 ==0):
		print (losses)

var  = tf.trainable_variables()
var = [v.name for v in tf.trainable_variables() ]
print (var)
numpy_val = sess.run([var[0]])
weights = np.asarray(numpy_val[0])
print (weights)

print ("Observe that weights are close to [0.299,0.587,0.114] , hence it learns rgb to gray conversion")








