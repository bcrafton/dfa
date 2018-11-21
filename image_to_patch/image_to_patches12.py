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

image = tf.placeholder(tf.float32, [10, 30, 30, 3])
pad = tf.pad(tensor=image, paddings=[[0, 0], [2,2], [2,2], [0,0]], mode='CONSTANT')
filters = tf.placeholder(tf.float32, [1, 100, 3, 3, 3, 64])

slice1 = tf.slice(pad, begin=[0, 0, 0, 0], size=[10, 30, 30, 3])
slice2 = tf.slice(pad, begin=[0, 0, 1, 0], size=[10, 30, 30, 3])
slice3 = tf.slice(pad, begin=[0, 0, 2, 0], size=[10, 30, 30, 3])
slices1 = tf.concat((slice1, slice2, slice3), axis=2)

slice4 = tf.slice(pad, begin=[0, 1, 0, 0], size=[10, 30, 30, 3])
slice5 = tf.slice(pad, begin=[0, 1, 1, 0], size=[10, 30, 30, 3])
slice6 = tf.slice(pad, begin=[0, 1, 2, 0], size=[10, 30, 30, 3])
slices2 = tf.concat((slice4, slice5, slice6), axis=2)

slice7 = tf.slice(pad, begin=[0, 2, 0, 0], size=[10, 30, 30, 3])
slice8 = tf.slice(pad, begin=[0, 2, 1, 0], size=[10, 30, 30, 3])
slice9 = tf.slice(pad, begin=[0, 2, 2, 0], size=[10, 30, 30, 3])
slices3 = tf.concat((slice7, slice8, slice9), axis=2)

slices = tf.concat((slices1, slices2, slices3), axis=1)

patches = image_to_patches(slices, tile_size, tile_size)
y = tf.multiply(tf.reshape(patches, (-1, 100, 3, 3, 3, 1)), filters)
z = tf.reduce_sum(y, axis=(2, 3, 4))
z = tf.reshape(z, (-1, 30, 30, 64))

#########################################################

sess = tf.Session()

cifar10 = tf.keras.datasets.cifar10.load_data()
(x_train, y_train), (x_test, y_test) = cifar10
x_train = x_train.transpose(0, 2, 3, 1)
y_train = keras.utils.to_categorical(y_train, 10)
x_test = x_test.transpose(0, 2, 3, 1)
y_test = keras.utils.to_categorical(y_test, 10)

x_train = x_train[0:10]
x_train = x_train[:, 0:30, 0:30, :]
_image = x_train
 
_filters = np.random.uniform(low=1., high=1., size=(1, 100, 3, 3, 3, 64))
[_slices, _patches, _z] = sess.run([slices, patches, z], feed_dict={image: _image, filters: _filters})

print (np.shape(_slices), np.shape(_patches), np.shape(_z))

# img = x_train[4]
img = _z[4, :, :, 0]
img = img / np.max(img)
# plt.imsave('test.png', img)
plt.imshow(img)

plt.show()






