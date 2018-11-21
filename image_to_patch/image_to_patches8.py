import tensorflow as tf
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import keras
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
from tensorflow.python.ops.array_grad import _ExtractImagePatchesGrad

tile_size = 112

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

image = tf.placeholder(tf.float32, [None, 224, 224, 3])
pad = tf.pad(tensor=image, paddings=[[0, 0], [2,2], [2,2], [0,0]], mode='CONSTANT')
filters = tf.placeholder(tf.float32, [1, 50176, 3, 3, 3, 64])
slice1 = tf.slice(pad, begin=[0, 0, 0, 0], size=[1, 224, 224, 3])
slice2 = tf.slice(pad, begin=[0, 1, 0, 0], size=[1, 224, 224, 3])
slice3 = tf.slice(pad, begin=[0, 2, 0, 0], size=[1, 224, 224, 3])
slices1 = tf.concat((slice1, slice2, slice3), axis=1)
patches = image_to_patches(slices1, tile_size, tile_size)
patches = tf.reshape(patches, (1, 12 * tile_size, tile_size, 3))


#########################################################

sess = tf.Session()

cifar10 = tf.keras.datasets.cifar10.load_data()
(x_train, y_train), (x_test, y_test) = cifar10

_image = Image.open('laska.png')
_image.load()
_image = np.asarray(_image, dtype="float32")
_image = _image[0:224, 0:224, 0:3]
shape = np.shape(_image)
_image = np.reshape(_image, (1, shape[0], shape[1], shape[2]))
    
_filters = np.random.uniform(low=0.5, high=1.0, size=(1, 50176, 3, 3, 3, 64))
[_slices1, _patches] = sess.run([slices1, patches], feed_dict={image: _image, filters: _filters})

print (np.shape(_slices1), np.shape(_patches))

plt.imshow(_patches[0] / np.max(_patches[0]))
plt.show()






