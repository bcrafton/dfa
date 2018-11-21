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
filters = tf.placeholder(tf.float32, [1, 1, 3, 3, 3, 64])

slice1 = tf.slice(pad, begin=[0, 0, 0, 0], size=[1, 9, 9, 3])
slice2 = tf.slice(pad, begin=[0, 0, 1, 0], size=[1, 9, 9, 3])
slice3 = tf.slice(pad, begin=[0, 0, 2, 0], size=[1, 9, 9, 3])
slices1 = tf.concat((slice1, slice2, slice3), axis=2)

slice4 = tf.slice(pad, begin=[0, 1, 0, 0], size=[1, 9, 9, 3])
slice5 = tf.slice(pad, begin=[0, 1, 1, 0], size=[1, 9, 9, 3])
slice6 = tf.slice(pad, begin=[0, 1, 2, 0], size=[1, 9, 9, 3])
slices2 = tf.concat((slice4, slice5, slice6), axis=2)

slice7 = tf.slice(pad, begin=[0, 2, 0, 0], size=[1, 9, 9, 3])
slice8 = tf.slice(pad, begin=[0, 2, 1, 0], size=[1, 9, 9, 3])
slice9 = tf.slice(pad, begin=[0, 2, 2, 0], size=[1, 9, 9, 3])
slices3 = tf.concat((slice7, slice8, slice9), axis=2)

slices = tf.concat((slices1, slices2, slices3), axis=1)

patches = image_to_patches(slices, tile_size, tile_size)
# patches = tf.reshape(patches, (1, 81 * tile_size, tile_size, 3))


#########################################################

sess = tf.Session()

cifar10 = tf.keras.datasets.cifar10.load_data()
(x_train, y_train), (x_test, y_test) = cifar10

# _image = Image.open('laska.png')
# _image.load()
# _image = np.asarray(_image, dtype="float32")
# _image = _image[0:224, 0:224, 0:3]
_image = np.linspace(0.1, 1.0, 81 * 3)
_image = np.reshape(_image, (1, 9, 9, 3))
# shape = np.shape(_image)
# _image = np.reshape(_image, (1, shape[0], shape[1], shape[2]))
    
_filters = np.random.uniform(low=0.5, high=1.0, size=(1, 1, 3, 3, 3, 64))
[_pad, _slices, _patches] = sess.run([pad, slices, patches], feed_dict={image: _image, filters: _filters})

'''
print (np.shape(_slices), np.shape(_patches))

plt.imsave('test.png', _slices[0] / np.max(_slices[0]))
plt.imshow(_slices[0] / np.max(_slices[0]))

plt.imsave('test.png', _patches[0] / np.max(_patches[0]))
plt.imshow(_patches[0] / np.max(_patches[0]))

plt.show()
'''

_patches = np.reshape(_patches, (81, 3, 3, 3))
print (np.shape(_pad), np.shape(_patches))


_patches2 = np.zeros(shape=(81, 3, 3, 3))
for ii in range(9):
    for jj in range(9):
        _patches2[ii * 9 + jj] = _pad[0, ii:ii+3, jj:jj+3, :]

print(np.sum(_patches - _patches2))

print (_patches)
print (_patches2)






