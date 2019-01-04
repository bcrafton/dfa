import tensorflow as tf
import numpy as np
# from skimage import io
import matplotlib.pyplot as plt
import keras
from PIL import Image
np.set_printoptions(threshold=np.inf)

#########################################################

image = tf.placeholder(tf.float32, [None, 225, 225, 3])
kernel = tf.Variable(tf.random_uniform(shape=[75*75, 3*3*3, 32], minval=1., maxval=1.))
error = tf.Variable(tf.random_uniform(shape=[1, 75, 75, 32], minval=1., maxval=1.))

#########################################################

# output = K.local_conv2d(inputs, self.kernel, self.kernel_size, self.strides, (self.output_row, self.output_col))
# kernel: the unshared weight for convolution, with shape (output_items, feature_dim, filters)
# self.kernel_shape = ( output_row * output_col, self.kernel_size[0] * self.kernel_size[1] * input_filter, self.filters)

def local_conv2d(inputs, kernel, kernel_size, strides, output_shape):

  N = tf.shape(inputs)[0]

  stride_row, stride_col = strides
  output_row, output_col = output_shape
  kernel_shape = tf.shape(kernel)
  feature_dim = kernel_shape[1]
  filters = kernel_shape[2]

  xs = []
  for i in range(output_row):
    for j in range(output_col):
      slice_row = slice(i * stride_row, i * stride_row + kernel_size[0])
      slice_col = slice(j * stride_col, j * stride_col + kernel_size[1])
      xs.append(tf.reshape(inputs[:, slice_row, slice_col, :], (1, N, feature_dim)))

  x_aggregate = tf.concat(xs, axis=0)
  output = tf.keras.backend.batch_dot(x_aggregate, kernel)
  output = tf.reshape(output, (output_row, output_col, N, filters))
  output = tf.transpose(output, (2, 0, 1, 3))
  
  return x_aggregate, output

x_aggregate, output = local_conv2d(image, kernel, [3, 3], strides=[3, 3], output_shape=[75, 75])

# AI = tf.reshape(x_aggregate, (1, 75*75, 1, 27, 1))

# DO = tf.reshape(error,       (1, 75*75, 1, 1, 32))

# this should probably use that 'batch_dot'
# think about similarities in compute.  
# DW = tf.multiply(AI, DO)

#########################################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()

#####################################################

_image = Image.open('laska.png')
_image.load()
_image = np.asarray(_image, dtype="float32")
_image = _image[0:225, 0:225, 0:3]
_shape = np.shape(_image)
_image = np.reshape(_image, (1, _shape[0], _shape[1], _shape[2]))
_image = np.concatenate((_image, _image, _image, _image), axis=0)

[_x_aggregate, _kernel, _output] = sess.run([x_aggregate, kernel, output], feed_dict={image: _image})

print (np.shape(_x_aggregate), np.shape(_kernel), np.shape(_output))

img = _output[0, :, :, 0] / np.max(_output[0, :, :, 0])
plt.imshow(img)
plt.show()

#####################################################





