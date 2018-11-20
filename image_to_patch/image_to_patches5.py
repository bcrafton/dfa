import tensorflow as tf
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import keras
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image

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
filters = tf.placeholder(tf.float32, [1, 5625, 3, 3, 3, 64])

patches = image_to_patches(image, tile_size, tile_size)
y = tf.multiply(tf.reshape(patches, (-1, 5625, 3, 3, 3, 1)), filters)
z = tf.reduce_sum(y, axis=(2, 3, 4))
z = tf.reshape(z, (-1, 75, 75, 64))
inv = patches_to_image(patches, 75, 75)

#########################################################

sess = tf.Session()

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
    
_filters = np.random.uniform(low=1.0, high=1.0, size=(1, 5625, 3, 3, 3, 64))
_image, _patches, _y, _z, _inv = sess.run([image, patches, y, z, inv], feed_dict={image: _image, filters: _filters})

print('image', _image.shape)
print('patches', _patches.shape)
print('y', _y.shape)
print('z', _z.shape)
print('inv', _inv.shape)

plt.imshow(_z[0, :, :, 0] / np.max(_z[0, :, :, 0]))
plt.show()



