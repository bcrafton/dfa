import tensorflow as tf
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import keras
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
from tensorflow.python.ops.array_grad import _ExtractImagePatchesGrad
np.set_printoptions(threshold=np.inf)

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

image = tf.placeholder(tf.float32, [1, 30, 30, 3])
pad = tf.pad(tensor=image, paddings=[[0, 0], [2,2], [2,2], [0,0]], mode='CONSTANT')
filters = tf.placeholder(tf.float32, [1, 100, 3, 3, 3, 64])

slice1 = tf.slice(pad, begin=[0, 0, 0, 0], size=[1, 30, 30, 3])
patch1 = image_to_patches(slice1, tile_size, tile_size)

slice2 = tf.slice(pad, begin=[0, 0, 1, 0], size=[1, 30, 30, 3])
patch2 = image_to_patches(slice2, tile_size, tile_size)

slice3 = tf.slice(pad, begin=[0, 0, 2, 0], size=[1, 30, 30, 3])
patch3 = image_to_patches(slice3, tile_size, tile_size)

slices1 = tf.concat((patch1, patch2, patch3), axis=2)
patches = image_to_patches(slices1, tile_size, tile_size)

#########################################################

sess = tf.Session()

cifar10 = tf.keras.datasets.cifar10.load_data()
(x_train, y_train), (x_test, y_test) = cifar10
x_train = x_train.transpose(0, 2, 3, 1)
y_train = keras.utils.to_categorical(y_train, 10)
x_test = x_test.transpose(0, 2, 3, 1)
y_test = keras.utils.to_categorical(y_test, 10)

x_train = x_train[0:1]
x_train = x_train[:, 0:30, 0:30, :]
_image = x_train
 
_filters = np.random.uniform(low=1., high=1., size=(1, 100, 3, 3, 3, 64))
[_slices1, _patches] = sess.run([slices1, patches], feed_dict={image: _image, filters: _filters})

print (np.shape(_slices1), np.shape(_patches))

'''
img = _patches
img = img / np.max(img)
plt.imshow(img)

plt.show()
'''





