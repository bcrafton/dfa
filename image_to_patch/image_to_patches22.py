import tensorflow as tf
import numpy as np
from skimage import io
import matplotlib.pyplot as plt
import keras
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image
from tensorflow.python.ops.array_grad import _ExtractImagePatchesGrad
np.set_printoptions(threshold=np.inf)

def image_to_patches(image, kernel_size, kernel_stride):
    shape = tf.shape(image)
    N = shape[0]
    height = shape[1]
    width = shape[2]
    channels = shape[3]
    kh, kw = kernel_size
    sh, sw = kernel_stride
    
    print (height, width)
    
    # assert stride height = stride width
    assert(sh == sw)
    # assert stride height = kernel height or 1
    assert(sh == kh or sh == 1)
    # assert stride width = kernel width or 1
    assert(sw == kw or sw == 1)
    # assert kernel width = kernel height
    assert(kh == kw)
    # assert kh and divides height
    # assert((height % kh) == 0)
    # assert kw and divides width
    # assert((width % kw) == 0)
    
    rows = height // kh
    cols = width // kw
   
    if (sh == 1):
        sh, sw = kh, kw # stride = slice now.
        num_slices = sh * sw
    
        pad = tf.pad(tensor=image, paddings=[[0, 0], [kh-1,kh-1], [kw-1,kw-1], [0,0]], mode='CONSTANT')
    
        slices = []
        for ii in range(sh):
            for jj in range(sw):
                slic = tf.reshape(tf.slice(pad, begin=[0, ii, jj, 0], size=[1, 9, 9, 3]), (1, 1, 9, 9, 3))
                slices.append(slic)
                
        slices = tf.concat(slices, axis=1)
        
        # [1, 9, 9, 9, 3]
        slices = tf.reshape(slices, [N, num_slices, rows, kh, width, channels])
        # [1, 9, 3, 3, 9, 3]
        slices = tf.transpose(slices, [0, 1, 2, 4, 3, 5])
        # [1, 9, 3, 9, 3, 3]
        slices = tf.reshape(slices, [N, num_slices, rows * cols, kw, kh, channels])
        # [1, 9, 9, 3, 3, 3]
        slices = tf.transpose(slices, [0, 1, 2, 4, 3, 5])
        # [1, 9, 9, 3, 3, 3]
        
        # [1, 9, 9, 3, 3, 3]
        slices = tf.reshape(slices, [N, sh, sw, rows * cols, kh, kw, channels])
        # [1, 3, 3, 9, 3, 3, 3]
        slices = tf.transpose(slices, [0, 1, 3, 2, 4, 5, 6])
        # [1, 3, 9, 3, 3, 3, 3]
        slices = tf.reshape(slices, [N, sh, rows, cols * sw, kh, kw, channels])
        # [1, 3, 3, 9, 3, 3, 3]
        slices = tf.transpose(slices, [0, 2, 1, 3, 4, 5, 6])
        # [1, 3, 3, 3, 27, 3]
        slices = tf.reshape(slices, [N, rows * sh * cols * sw, kh, kw, channels])

        return slices

    else:
        # [1, 9, 9, 3]
        patches = tf.reshape(image, [N, rows, kh, width, channels])
        # [1, 3, 3, 9, 3]
        patches = tf.transpose(patches, [0, 1, 3, 2, 4])
        # [1, 3, 9, 3, 3]
        patches = tf.reshape(patches, [N, rows * cols, kw, kh, channels])
        # [1, 9, 3, 3, 3]
        patches = tf.transpose(patches, [0, 1, 3, 2, 4])
        
        return patches

#########################################################

image = tf.placeholder(tf.float32, [None, 9, 9, 3])
pad = tf.pad(tensor=image, paddings=[[0, 0], [2,2], [2,2], [0,0]], mode='CONSTANT')
slices = image_to_patches(image, (3, 3), (1, 1))

#########################################################

sess = tf.Session()

_image = np.linspace(0.1, 1.0, 1 * 9 * 9 * 3)
_image = np.reshape(_image, (1, 9, 9, 3))

#####################################################

[_pad, _patches] = sess.run([pad, slices], feed_dict={image: _image})

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




