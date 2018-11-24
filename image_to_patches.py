import tensorflow as tf
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import keras
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image

from BioConvolution import BioConvolution
from Activation import Linear

tile_size = 3

#########################################################

image = tf.placeholder(tf.float32, [10, 30, 30, 3])

l1 = BioConvolution(input_sizes=[10, 30, 30, 3], filter_sizes=[3, 3, 3, 64], num_classes=10, init_filters='sqrt_fan_in', strides=[1, 1, 1, 1], padding='SAME', alpha=0.01, activation=Linear(), bias=0., last_layer=False, name='conv1')
l1_forward = l1.forward(image)
l1_backward = l1.backward(image, l1_forward, l1_forward)
l1_gv = l1.gv(image, l1_forward, l1_forward)

#########################################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

cifar10 = tf.keras.datasets.cifar10.load_data()
(x_train, y_train), (x_test, y_test) = cifar10
x_train = x_train.transpose(0, 2, 3, 1)
y_train = keras.utils.to_categorical(y_train, 10)
x_test = x_test.transpose(0, 2, 3, 1)
y_test = keras.utils.to_categorical(y_test, 10)
x_train = x_train[0:10]
x_train = x_train[:, 0:30, 0:30, :]

[_l1_forward, _l1_backward, _l1_gv] = sess.run([l1_forward, l1_backward, l1_gv], feed_dict={image: x_train})

print (np.shape(_l1_forward), np.shape(_l1_backward), np.shape(_l1_gv[0][0]), np.shape(_l1_gv[1][0]))

img = _l1_forward[0, :, :, 0] / np.max(_l1_forward[0, :, :, 0])
shape = np.shape(img)
plt.imshow(img)
plt.show()



