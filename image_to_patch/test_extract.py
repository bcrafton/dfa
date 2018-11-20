
import argparse
import os
import sys

import time
import tensorflow as tf
import keras
import math
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

##############################################

cifar10 = tf.keras.datasets.cifar10.load_data()

##############################################

X = tf.placeholder(tf.float32, [None, 32, 32, 3])
F = tf.placeholder(tf.float32, [1, 121, 27, 64])

img = tf.extract_image_patches(images=X, ksizes=[1, 3, 3, 1], strides=[1, 3, 3, 1], rates=[1, 1, 1, 1], padding='SAME')
shape = tf.shape(img)
img = tf.reshape(img, (shape[0], shape[1] * shape[2], shape[3], 1))

Y = tf.multiply(img, F)
O = tf.reduce_sum(Y, axis=2)

##############################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()

(x_train, y_train), (x_test, y_test) = cifar10

##############################################

n = 3
x = np.reshape(x_train[0:n], (n, 32, 32, 3))
f = np.random.uniform(size=(1, 121, 27, 64))
[_img, _Y] = sess.run([img, Y], feed_dict={X: x, F: f})
print (np.shape(_img), np.shape(_Y), np.shape(O))

###

# so 3x121x27 * 121x27 works good
# but we want 3x121x27 * 121*27*64 as well.
# can look here: 
# https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html
# bc tf follows it ...
# https://www.tensorflow.org/api_docs/python/tf/math/multiply
# but dont think ull get both.
# wow it works!
# Two dimensions are compatible when
# - they are equal, or
# - one of them is 1















