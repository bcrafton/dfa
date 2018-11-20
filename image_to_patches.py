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

def image_to_patches(image, patch_height, patch_width):
    shape = tf.shape(image)
    N = shape[0]
    img_height = shape[1]
    img_width = shape[2]
    channels = shape[3]
    # image passed has to be divisible by the patch.
    height = img_height
    # image passed has to be divisible by the patch.
    width = img_width
    num_rows = height // patch_height
    num_cols = width // patch_width
    
    img = tf.reshape(image, (N, height, width, channels))
    
    img = tf.reshape(img, [N, num_rows, patch_height, width, -1])
    img = tf.transpose(img, [0, 1, 3, 2, 4])
    img = tf.reshape(img, [N, num_rows, num_cols, patch_width, patch_height, -1])
    img = tf.reshape(img, [N, num_rows * num_cols, patch_width, patch_height, -1])
    img = tf.transpose(img, [0, 1, 3, 2, 4])
    
    return img
    
def patches_to_image(patches, img_height, img_width):
    shape = tf.shape(patches)
    N = shape[0]
    num_patches = shape[1]
    patch_height = shape[2]
    patch_width = shape[3]
    channels = shape[4]
    # image passed has to be divisible by the patch.
    height = img_height
    # image passed has to be divisible by the patch.
    width = img_width
    num_rows = height // patch_height
    num_cols = width // patch_width

    img = tf.transpose(patches, [0, 1, 3, 2, 4])
    img = tf.reshape(img, [N, num_rows, num_cols * patch_width, patch_height, -1])
    img = tf.transpose(img, [0, 1, 3, 2, 4])
    img = tf.reshape(img, [N, num_rows * patch_height, width, -1])
    
    return img

#########################################################

image = tf.placeholder(tf.float32, [None, 225, 225, 3])
# filters1 = tf.placeholder(tf.float32, [1, 5625, 3, 3, 3, 64])
# filters2 = tf.placeholder(tf.float32, [1, 625, 3, 3, 64, 128])

l1 = BioConvolution(input_sizes=[1, 225, 225, 3], filter_sizes=[3, 3, 3, 64], num_classes=10, init_filters='sqrt_fan_in', strides=[1, 3, 3, 1], padding='SAME', alpha=0.01, activation=Linear(), bias=0., last_layer=False, name='conv1')
l1_forward = l1.forward(image)
l1_backward = l1.backward(image, l1_forward, l1_forward)
l1_grad = l1.gv(image, l1_forward, l1_forward)

#########################################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

cifar10 = tf.keras.datasets.cifar10.load_data()
(x_train, y_train), (x_test, y_test) = cifar10

# print (np.shape(x_train))
# (50000, 3, 32, 32)
_image = Image.open('laska.png')
_image.load()
_image = np.asarray(_image, dtype="float32")
# tried 1:4 ... get a blue ferret.
# guess #4 = alpha
# 225 is divisible by 3
_image = _image[0:225, 0:225, 0:3]
shape = np.shape(_image)
_image = np.reshape(_image, (1, shape[0], shape[1], shape[2]))
    
# _filters1 = np.random.uniform(low=0.5, high=1.0, size=(1, 5625, 3, 3, 3, 64))
# _filters2 = np.random.uniform(low=0.5, high=1.0, size=(1, 625, 3, 3, 64, 128))
# _z, _z2, _dz = sess.run([z, z2, dz], feed_dict={image: _image, filters1: _filters1, filters2: _filters2})
_l1_forward, _l1_backward, _l1_grad = sess.run([l1_forward, l1_backward, l1_grad], feed_dict={image: _image})

print ( np.shape(_l1_forward), np.shape(_l1_backward), np.shape(_l1_grad[0][0]), np.shape(_l1_grad[0][1]), np.shape(_l1_grad[1][0]), np.shape(_l1_grad[1][1]) )

plt.imshow(_l1_backward[0, :, :, :] / np.max(_l1_backward[0, :, :, :]))
plt.show()



