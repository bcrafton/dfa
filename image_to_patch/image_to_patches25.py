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
num_rows = 3
num_cols = 3
patch_height = 3
patch_width = 3
width = 9
height = 9
channels = 3

#########################################################

image = tf.placeholder(tf.float32, [None, 9, 9, 3])
pad = tf.pad(tensor=image, paddings=[[0, 0], [2,2], [2,2], [0,0]], mode='CONSTANT')

#########################################################

slice1 = tf.reshape(tf.slice(pad, begin=[0, 0, 0, 0], size=[1, 9, 9, 3]), (1, 1, 9, 9, 3))
slice2 = tf.reshape(tf.slice(pad, begin=[0, 0, 1, 0], size=[1, 9, 9, 3]), (1, 1, 9, 9, 3))
slice3 = tf.reshape(tf.slice(pad, begin=[0, 0, 2, 0], size=[1, 9, 9, 3]), (1, 1, 9, 9, 3))
slice4 = tf.reshape(tf.slice(pad, begin=[0, 1, 0, 0], size=[1, 9, 9, 3]), (1, 1, 9, 9, 3))
slice5 = tf.reshape(tf.slice(pad, begin=[0, 1, 1, 0], size=[1, 9, 9, 3]), (1, 1, 9, 9, 3))
slice6 = tf.reshape(tf.slice(pad, begin=[0, 1, 2, 0], size=[1, 9, 9, 3]), (1, 1, 9, 9, 3))
slice7 = tf.reshape(tf.slice(pad, begin=[0, 2, 0, 0], size=[1, 9, 9, 3]), (1, 1, 9, 9, 3))
slice8 = tf.reshape(tf.slice(pad, begin=[0, 2, 1, 0], size=[1, 9, 9, 3]), (1, 1, 9, 9, 3))
slice9 = tf.reshape(tf.slice(pad, begin=[0, 2, 2, 0], size=[1, 9, 9, 3]), (1, 1, 9, 9, 3))
slices = tf.concat((slice1, slice2, slice3, slice4, slice5, slice6, slice7, slice8, slice9), axis=1)

# [1, 9, 9, 9, 3]
slices = tf.reshape(slices, [1, 9, 3, 3, 9, 3])
# [1, 9, 3, 3, 9, 3]
slices = tf.transpose(slices, [0, 1, 2, 4, 3, 5])
# [1, 9, 3, 9, 3, 3]
slices = tf.reshape(slices, [1, 9, 9, 3, 3, 3])
# [1, 9, 9, 3, 3, 3]
slices = tf.transpose(slices, [0, 1, 2, 4, 3, 5])
# [1, 9, 9, 3, 3, 3]

slices = tf.reshape(slices, [1, 3, 3, 9, 3, 3, 3])
# [1, 3, 3, 9, 3, 3, 3]
slices = tf.transpose(slices, [0, 1, 3, 2, 4, 5, 6])
# [1, 3, 9, 3, 3, 3, 3]
slices = tf.reshape(slices, [1, 3, 3, 9, 3, 3, 3])
# [1, 3, 3, 9, 3, 3, 3]
slices = tf.transpose(slices, [0, 2, 1, 3, 4, 5, 6])
# [1, 3, 3, 3, 27, 3]
slices = tf.reshape(slices, [1, 27, 27, 3])
# [1, 27, 27, 3]
patches = slices

#########################################################

sess = tf.Session()

_image = np.linspace(0.1, 1.0, 1 * 9 * 9 * 3)
_image = np.reshape(_image, (1, 9, 9, 3))

#####################################################

[_pad, _slices, _patches] = sess.run([pad, slices, patches], feed_dict={image: _image})

#####################################################

print (np.shape(_slices))

#####################################################

plt.subplot(1, 2, 1)
_patches = np.reshape(_patches, (1, 81, 9, 3))
plt.imshow(_patches[0] / np.max(_patches[0]))

_patches2 = np.zeros(shape=(1, 81, 9, 3))
for ii in range(9):
    for jj in range(9):
        _patches2[0, ii * 9 + jj, :, :] = np.reshape(_pad[0, ii:(ii+3), jj:(jj+3), :], (9, 3))

plt.subplot(1, 2, 2)
_patches2 = np.reshape(_patches2, (1, 81, 9, 3))
plt.imshow(_patches2[0] / np.max(_patches2[0]))

plt.show()




