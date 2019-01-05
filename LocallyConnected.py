
import tensorflow as tf
import numpy as np
import math

import conv_utils

from Layer import Layer
   
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
    return output
  
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
    img = tf.reshape(img, [N, num_rows, num_cols * patch_width, patch_height, channels])
    img = tf.transpose(img, [0, 1, 3, 2, 4])
    img = tf.reshape(img, [N, num_rows * patch_height, width, channels])
    
    return img
    
def get_pad(padding, filter_size):

  if padding == 'same':
      pad = filter_size // 2
      
  elif padding == 'valid':
      pad = 0

  elif padding == 'full':
      pad = filter_size - 1
      
  else:
      assert(False)

  return pad
  
class LocallyConnected(Layer):

    def __init__(self, input_size, filter_size, num_classes, init, strides, padding, activation, bias, last_layer, name=None, load=None, train=True):
        self.input_size = input_size
        self.filter_size = filter_size
        self.num_classes = num_classes
        self.batch_size, self.h, self.w, self.fin = self.input_size
        self.fh, self.fw, self.fin, self.fout = self.filter_size
        self.bias = tf.Variable(tf.ones(shape=self.fout) * bias)
        self.strides = strides
        self.stride_row, self.stride_col = self.strides
        self.padding = padding
        self.activation = activation
        self.last_layer = last_layer
        self.name = name
        self._train = train
        
        self.alpha = 0.01

        self.output_row = conv_utils.conv_output_length(self.h, self.fh, self.padding, self.strides[0])
        self.output_col = conv_utils.conv_output_length(self.w, self.fw, self.padding, self.strides[1])
        self.output_shape = (self.output_row, self.output_col)
        self.filter_shape = (self.output_row * self.output_col, self.fh * self.fw * self.fin, self.fout)
        
        print (self.name + ' input shape: ' + str(self.input_size))
        print (self.name + ' filter shape: ' + str(self.filter_shape))
        print (self.name + ' output shape: ' + str(self.output_shape))
        
        if init == "zero":
            self.filters = tf.Variable(tf.zeros(shape=self.filter_shape), dtype=tf.float32)
        elif init == "sqrt_fan_in":
            sqrt_fan_in = math.sqrt(self.h*self.w*self.fin)
            self.filters = tf.Variable(tf.random_uniform(shape=self.filter_shape, minval=-1.0/sqrt_fan_in, maxval=1.0/sqrt_fan_in), dtype=tf.float32)
        elif init == "alexnet":
            self.filters = tf.Variable(np.random.normal(loc=0.0, scale=0.01, size=self.filter_shape), dtype=tf.float32)
        else:
            self.filters = tf.get_variable(name=self.name, shape=self.filter_shape)

    ###################################################################

    def get_weights(self):
        return [(self.name, self.filters), (self.name + "_bias", self.bias)]

    def num_params(self):
        filter_weights_size = np.prod(self.filter_shape)
        bias_weights_size = self.fout
        return filter_weights_size + bias_weights_size
                
    def forward(self, X):
        # return local_conv2d(X, self.filters, (self.fh, self.fw), self.strides, (self.output_row, self.output_col))
        
        N = tf.shape(X)[0]
    
        xs = []
        for i in range(self.output_row):
            for j in range(self.output_col):
                slice_row = slice(i * self.stride_row, i * self.stride_row + self.fh)
                slice_col = slice(j * self.stride_col, j * self.stride_col + self.fw)
                xs.append(tf.reshape(X[:, slice_row, slice_col, :], (1, N, self.fh * self.fw * self.fin)))

        x_aggregate = tf.concat(xs, axis=0) 
        output = tf.keras.backend.batch_dot(x_aggregate, self.filters)
        output = tf.reshape(output, (self.output_row, self.output_col, N, self.fout))
        output = tf.transpose(output, (2, 0, 1, 3))
        return output
        
    ###################################################################           

    # This operation pads a tensor according to the paddings you specify. 
    # paddings is an integer tensor with shape [n, 2], where n is the rank of tensor. 
    # For each dimension D of input, paddings[D, 0] indicates how many values to add 
      # before the contents of tensor in that dimension, and paddings[D, 1] indicates 
      # how many values to add after the contents of tensor in that dimension.

    def backward(self, AI, AO, DO):
    
        N = tf.shape(AI)[0]
        
        [pad_w, pad_h] = get_pad('full', np.array([self.fh, self.fw]))
        DO = tf.pad(DO, [[0, 0], [pad_w, pad_w], [pad_h, pad_h], [0, 0]])
        
        es = []
        for i in range(self.output_row):
            for j in range(self.output_col):
                slice_row = slice(i, i + self.fh)
                slice_col = slice(j, j + self.fw)
                es.append(tf.reshape(DO[:, slice_row, slice_col, :], (1, N, self.fh * self.fw * self.fout)))
        
        DI = tf.concat(es, axis=0)
        
        # DI =      [900, 4, 288]
        # filters = [900, 27, 32]
        # filters = [900, 288, 3]
        # we change filters bc think batch_dot needs to be done (DI, F).
        
        filters = self.filters
        filters = tf.reshape(filters, (self.output_row * self.output_col, self.fh * self.fw, self.fin, self.fout))
        filters = tf.transpose(filters, (0, 1, 3, 2))
        filters = tf.reshape(filters, (self.output_row * self.output_col, self.fh * self.fw * self.fout, self.fin))
        
        DI = tf.keras.backend.batch_dot(DI, filters)
        
        return DI

    def gv(self, AI, AO, DO): 
        assert(False)

    def train(self, AI, AO, DO): 
        return []

    ###################################################################

    def dfa_backward(self, AI, AO, E, DO):
        pass
        
    def dfa_gv(self, AI, AO, E, DO):
        pass
        
    def dfa(self, AI, AO, E, DO): 
        pass
        
    ###################################################################    
        
        
        
        
