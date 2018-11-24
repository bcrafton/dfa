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

image = tf.placeholder(tf.float32, [9, 9])
pad = tf.pad(tensor=image, paddings=[[2,2], [2,2]], mode='CONSTANT')

#########################################################

sess = tf.Session()

_image = np.linspace(0, 9 * 9 - 1, 9 * 9)
_image = np.reshape(_image, (9, 9))

#####################################################

[_pad] = sess.run([pad], feed_dict={image: _image})

#####################################################

_patches2 = np.zeros(shape=(81, 9))
for ii in range(9):
    for jj in range(9):
        _patches2[ii * 9 + jj, :] = _pad[ii:(ii+3), jj:(jj+3)].flatten()
        print (_pad[ii:(ii+3), jj:(jj+3)])
    print ()


_patches2 = np.reshape(_patches2, (81, 9))
plt.subplot(1, 2, 2)
plt.imshow(_patches2 / np.max(_patches2))

plt.show()




