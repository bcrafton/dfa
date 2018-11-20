import tensorflow as tf
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import keras
from tensorflow.examples.tutorials.mnist import input_data

tile_size = 3

def image_to_patches(image, patch_height, patch_width):

    image_height = tf.cast(tf.shape(image)[1], dtype=tf.float32)
    image_width = tf.cast(tf.shape(image)[2], dtype=tf.float32)
    height = tf.cast(tf.ceil(image_height / patch_height) * patch_height, dtype=tf.int32)
    width = tf.cast(tf.ceil(image_width / patch_width) * patch_width, dtype=tf.int32)
    num_rows = height // patch_height
    num_cols = width // patch_width

    shape = tf.shape(image)
    N = shape[0]
    H = shape[1]
    W = shape[2]
    C = shape[3]

    img = tf.reshape(image, (N, H, W, C))
    img = tf.squeeze(tf.image.resize_image_with_crop_or_pad(img, height, width))
    
    # get slices along the 0-th axis
    img = tf.reshape(img, [N, num_rows, patch_height, width, -1])
    # h/patch_h, w, patch_h, c
    img = tf.transpose(img, [0, 1, 3, 2, 4])
    # get slices along the 1-st axis
    # h/patch_h, w/patch_w, patch_w,patch_h, c
    img = tf.reshape(img, [N, num_rows, num_cols, patch_width, patch_height, -1])
    # num_patches, patch_w, patch_h, c
    img = tf.reshape(img, [N, num_rows * num_cols, patch_width, patch_height, -1])
    # num_patches, patch_h, patch_w, c
    img = tf.transpose(img, [0, 1, 3, 2, 4])

    ###########################
    
    # num_patches, patch_w, patch_h, c
    img = tf.transpose(img, [0, 1, 3, 2, 4])
    
    img = tf.reshape(img, [N, num_rows, num_cols * patch_width, patch_height, -1])

    img = tf.transpose(img, [0, 1, 3, 2, 4])
    
    img = tf.reshape(img, [N, num_rows * patch_height, width, -1])
    
    return img

#########################################################

image = tf.placeholder(tf.float32, [None, 32, 32, 3])
tiles = image_to_patches(image, tile_size, tile_size)

#########################################################

sess = tf.Session()

cifar10 = tf.keras.datasets.cifar10.load_data()
(x_train, y_train), (x_test, y_test) = cifar10
# (50000, 3, 32, 32)
x_train = x_train.transpose(0, 2, 3, 1)
y_train = keras.utils.to_categorical(y_train, 10)
x_test = x_test.transpose(0, 2, 3, 1)
y_test = keras.utils.to_categorical(y_test, 10)

_image = np.reshape(x_train[0:10], (10, 32, 32, 3))
print('Original image shape:', image.shape)

_image, _tiles = sess.run([image, tiles], feed_dict={image: _image})

print(_image.shape)
print(_tiles.shape)

plt.imshow(_tiles[4] / np.max(_tiles[4]))
plt.show()



