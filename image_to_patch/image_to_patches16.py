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

image = tf.placeholder(tf.float32, [None, 9, 9, 3])
pad = tf.pad(tensor=image, paddings=[[0, 0], [2,2], [2,2], [0,0]], mode='CONSTANT')

#########################################################

slice1 = tf.slice(pad, begin=[0, 0, 0, 0], size=[1, 9, 9, 3])
patch1 = image_to_patches(slice1, 3, 3)

slice2 = tf.slice(pad, begin=[0, 0, 1, 0], size=[1, 9, 9, 3])
patch2 = image_to_patches(slice2, 3, 3)

slice3 = tf.slice(pad, begin=[0, 0, 2, 0], size=[1, 9, 9, 3])
patch3 = image_to_patches(slice3, 3, 3)

slices1 = tf.concat((patch1, patch2, patch3), axis=3)
patches1 = tf.reshape(slices1, (1, 9, 3, 9, 3))

#########################################################

slice4 = tf.slice(pad, begin=[0, 1, 0, 0], size=[1, 9, 9, 3])
patch4 = image_to_patches(slice4, 3, 3)

slice5 = tf.slice(pad, begin=[0, 1, 1, 0], size=[1, 9, 9, 3])
patch5 = image_to_patches(slice5, 3, 3)

slice6 = tf.slice(pad, begin=[0, 1, 2, 0], size=[1, 9, 9, 3])
patch6 = image_to_patches(slice6, 3, 3)

slices2 = tf.concat((patch4, patch5, patch6), axis=3)
patches2 = tf.reshape(slices2, (1, 9, 3, 9, 3))

#########################################################

slice7 = tf.slice(pad, begin=[0, 2, 0, 0], size=[1, 9, 9, 3])
patch7 = image_to_patches(slice7, 3, 3)

slice8 = tf.slice(pad, begin=[0, 2, 1, 0], size=[1, 9, 9, 3])
patch8 = image_to_patches(slice8, 3, 3)

slice9 = tf.slice(pad, begin=[0, 2, 2, 0], size=[1, 9, 9, 3])
patch9 = image_to_patches(slice9, 3, 3)

slices3 = tf.concat((patch7, patch8, patch9), axis=3)
patches3 = tf.reshape(slices3, (1, 9, 3, 9, 3))

#########################################################

patches1 = tf.transpose(patches1, (0, 1, 3, 2, 4))
patches1 = tf.reshape(patches1, (1, 3, 27, 3, 3)) 
patches1 = tf.transpose(patches1, (0, 1, 3, 2, 4))

patches2 = tf.transpose(patches2, (0, 1, 3, 2, 4))
patches2 = tf.reshape(patches2, (1, 3, 27, 3, 3)) 
patches2 = tf.transpose(patches2, (0, 1, 3, 2, 4))

patches3 = tf.transpose(patches3, (0, 1, 3, 2, 4))
patches3 = tf.reshape(patches3, (1, 3, 27, 3, 3)) 
patches3 = tf.transpose(patches3, (0, 1, 3, 2, 4))

patches = tf.concat((patches1, patches2, patches3), axis=2)
# patches = tf.reshape(patches, (1, 81, 3, 3, 3))

#########################################################

sess = tf.Session()

cifar10 = tf.keras.datasets.cifar10.load_data()
(x_train, y_train), (x_test, y_test) = cifar10

_image = np.linspace(0.1, 1.0, 1 * 9 * 9 * 3)
_image = np.reshape(_image, (1, 9, 9, 3))

#####################################################

[_pad, _patches, _patch1, _patch2, _patch3, _patches1] = sess.run([pad, patches, patch1, patch2, patch3, patches1], feed_dict={image: _image})

#####################################################

_patches2 = np.zeros(shape=(1, 27, 27, 3))
for ii in range(9):
    for jj in range(9):
        _patches2[0, ii*3:(ii+1)*3, jj*3:(jj+1)*3, :] = _pad[0, ii:ii+3, jj:jj+3, :]

#####################################################

print (np.shape(_patches))
print (np.shape(_patches2))

#####################################################

_patches = np.reshape(_patches, (1, 27, 27, 3))
# _patches = np.transpose(_patches, (0, 1, 3, 2, 4))
# _patches = np.reshape(_patches, (1, 81, 3, 3, 3))
# _patches = np.transpose(_patches, (0, 1, 3, 2, 4))

#####################################################

'''
_patches2 = np.copy(_patches2)
_patches2 = np.reshape(_patches2, (9, 3, 27, 3))
_patches2 = np.transpose(_patches2, (0, 2, 1, 3))
_patches2 = np.reshape(_patches2, (81, 3, 3, 3))
_patches2 = np.transpose(_patches2, (0, 2, 1, 3))
'''

#####################################################

print (np.sum(np.absolute(_patches - _patches2)))

plt.subplot(1, 2, 1)
_patches = np.reshape(_patches, (1, 27, 27, 3))
plt.imshow(_patches[0] / np.max(_patches[0]))

plt.subplot(1, 2, 2)
_patches2 = np.reshape(_patches2, (1, 27, 27, 3))
plt.imshow(_patches2[0] / np.max(_patches2[0]))


plt.show()


