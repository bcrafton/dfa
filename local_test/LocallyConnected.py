
import tensorflow as tf
import numpy as np
import math

from Layer import Layer
   
def local_conv2d(inputs, kernel, kernel_size, strides, output_shape):
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
            xs.append(tf.reshape(inputs[:, slice_row, slice_col, :], (1, -1, feature_dim)))

    x_aggregate = tf.concat(xs, axis=0)
    output = tf.keras.backend.batch_dot(x_aggregate, kernel)
    output = tf.reshape(output, (output_row, output_col, -1, filters))
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
  
class LocallyConnected(Layer):

    def __init__(self, input_size, filter_size, num_classes, init, strides, padding, activation, bias, last_layer, name=None, load=None, train=True):
        self.input_size = input_size
        self.filter_size = filter_size
        self.num_classes = num_classes
        self.batch_size, self.h, self.w, self.fin = self.input_sizes
        self.fh, self.fw, self.fin, self.fout = self.filter_sizes
        self.bias = tf.Variable(tf.ones(shape=self.fout) * bias)
        self.strides = strides
        self.stride_row, self.stride_col = self.strides
        self.padding = padding
        self.activation = activation
        self.last_layer = last_layer
        self.name = name
        self._train = train

        self.output_row = conv_utils.conv_output_length(self.h, self.fh, self.padding, self.strides[0])
        self.output_col = conv_utils.conv_output_length(self.w, self.fw, self.padding, self.strides[1])
        self.output_shape = (self.output_row, self.output_col)
        self.filter_shape = (self.output_row * self.output_col, self.fh * self.fw * self.fin, self.fout)
        
        if init_filters == "zero":
            self.filters = tf.Variable(tf.zeros(shape=self.filter_shape))
        elif init_filters == "sqrt_fan_in":
            sqrt_fan_in = math.sqrt(self.h*self.w*self.fin)
            self.filters = tf.Variable(tf.random_uniform(shape=self.filter_shape, minval=-1.0/sqrt_fan_in, maxval=1.0/sqrt_fan_in))
        elif init_filters == "alexnet":
            self.filters = tf.Variable(np.random.normal(loc=0.0, scale=0.01, size=self.filter_shape), dtype=tf.float32)
        else:
            self.filters = tf.get_variable(name=self.name, shape=self.filter_shape)

    ###################################################################

    def get_weights(self):
        return [(self.name, self.filters), (self.name + "_bias", self.bias)]

    def num_params(self):
        filter_weights_size = np.prod(filter_shape)
        bias_weights_size = self.fout
        return filter_weights_size + bias_weights_size
                
    def forward(self, X):
        return local_conv2d(X, self.filters, (self.fh, self.fw), self.strides, (self.output_row, self.output_col))
        
    ###################################################################           

    def backward(self, AI, AO, DO):
    
        # x_aggregate (5625, 4, 27)
        # output      (4, 75, 75, 32)
        # filters     (5625, 27, 32)
        N = tf.shape(AI)[0]
        
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DO = tf.transpose(DO, (1, 2, 0, 3))
        DO = tf.reshape(DO, (self.output_row, self.output_col, -1, filters))
        # DO          (5625, 4, 32)
        DO = tf.transpose(DO, (0, 2, 1))
        # DO          (5625, 32, 4)
        
        DI = tf.keras.backend.batch_dot(self.filters, DO)
        # DI          (5625, 27, 4)
        DI = tf.transpose(DI, (2, 0, 1))
        # DI          (4, 5625, 27)
        DI = tf.reshape(DI, (N, self.output_row, self.output_col, self.fh, self.fw, self.fin))
        # DI          (4, 75, 75, 3, 3, 3)

        DI = tf.transpose(DI, [0, 1, 3, 2, 4])
        DI = tf.reshape(DI, [N, self.output_row, self.output_col * self.fw, self.fh, self.fin])
        DI = tf.transpose(DI, [0, 1, 3, 2, 4])
        DI = tf.reshape(DI, [N, self.output_row * self.fh, self.w, self.fin])

        return DI

    '''
    def backward(self, AI, AO, DO):
        shape = tf.shape(AI)
        N = shape[0]
        height = shape[1]
        width = shape[2]
        channels = shape[3]

        # E
        # shape = (N, self.h // self.fh * self.w // self.fw, 1, 1, 1, self.fout)
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DO = tf.reshape(DO, (N, self.h // self.sh * self.w // self.sw, 1, 1, 1, self.fout))
        
        # F
        filters = tf.reshape(self.filters, self.filter_reshape)

        # DI
        DI = tf.multiply(DO, filters)
        if (self.sh == self.fh):
            DI = tf.reduce_sum(DI, axis=(5))
            DI = patches_to_image(DI, self.h, self.w)
        else:
            DI = tf.reduce_sum(DI, axis=(2, 3, 5))
            DI = tf.reshape(DI, (N, self.h // self.sh, self.w // self.sw, self.fin))

        # assert(False)

        return DI
    '''

    def gv(self, AI, AO, DO): 
        if not self._train:
            return []
    
        num_patches = self.output_row * self.output_col
        filter_size = self.fh * self.fw * self.fin

        xs = []
        for i in range(self.output_row):
            for j in range(self.output_col):
                slice_row = slice(i * self.stride_row, i * self.stride_row + self.fh)
                slice_col = slice(j * self.stride_col, j * self.stride_col + self.fw)
                xs.append(tf.reshape(AI[:, slice_row, slice_col, :], (1, -1, filter_size)))

        x_aggregate = tf.concat(xs, axis=0) 
        
        # x_aggregate (5625, 4, 27)
        # output      (4, 75, 75, 32)

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DO = tf.transpose(DO, (1, 2, 0, 3))
        DO = tf.reshape(DO, (output_row, output_col, -1, filters))
        # DO          (5625, 4, 32)
        
        x_aggregate = tf.transpose(DO, (0, 2, 1))
        # x_aggregate (5625, 27, 4)

        DF = tf.keras.backend.batch_dot(x_aggregate, DO)
        DB = tf.reduce_sum(DO, axis=[0, 1])

        return [(DF, self.filters), (DB, self.bias)]

    '''
    def gv(self, AI, AO, DO):  
        if not self._train:
            return []

        shape = tf.shape(AI)
        N = shape[0]
        height = shape[1]
        width = shape[2]
        channels = shape[3]

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DO = tf.reshape(DO, (N, self.h // self.sh * self.w // self.sw, 1, 1, 1, self.fout))
        
        _AI = image_to_patches(AI, (self.fh, self.fw), (self.sh, self.sw))
        shape = tf.shape(_AI)
        _AI = tf.reshape(_AI, (shape[0], shape[1], shape[2], shape[3], shape[4], 1))
        
        DF = tf.multiply(DO, _AI)
        DF = tf.reduce_sum(DF, axis=0)
        DB = tf.reduce_sum(DO, axis=[0, 1, 2, 3, 4])

        return [(DF, self.filters), (DB, self.bias)]
    '''

    def train(self, AI, AO, DO): 
        if not self._train:
            return []
    
        num_patches = self.output_row * self.output_col
        filter_size = self.fh * self.fw * self.fin

        xs = []
        for i in range(self.output_row):
            for j in range(self.output_col):
                slice_row = slice(i * self.stride_row, i * self.stride_row + self.fh)
                slice_col = slice(j * self.stride_col, j * self.stride_col + self.fw)
                xs.append(tf.reshape(AI[:, slice_row, slice_col, :], (1, -1, filter_size)))

        x_aggregate = tf.concat(xs, axis=0) 
        
        # x_aggregate (5625, 4, 27)
        # output      (4, 75, 75, 32)

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DO = tf.transpose(DO, (1, 2, 0, 3))
        DO = tf.reshape(DO, (output_row, output_col, -1, filters))
        # DO          (5625, 4, 32)
        
        x_aggregate = tf.transpose(DO, (0, 2, 1))
        # x_aggregate (5625, 27, 4)

        DF = tf.keras.backend.batch_dot(x_aggregate, DO)
        DB = tf.reduce_sum(DO, axis=[0, 1])

        self.filters = self.filters.assign(tf.subtract(self.filters, tf.scalar_mul(self.alpha, DF)))
        self.bias = self.bias.assign(tf.subtract(self.bias, tf.scalar_mul(self.alpha, DB)))
        return [(DF, self.filters), (DB, self.bias)]


    '''
    def train(self, AI, AO, DO): 
        if not self._train:
            return []

        shape = tf.shape(AI)
        N = shape[0]
        height = shape[1]
        width = shape[2]
        channels = shape[3]

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DO = tf.reshape(DO, (N, self.h // self.sh * self.w // self.sw, 1, 1, 1, self.fout))
        
        _AI = image_to_patches(AI, (self.fh, self.fw), (self.sh, self.sw))
        shape = tf.shape(_AI)
        _AI = tf.reshape(_AI, (shape[0], shape[1], shape[2], shape[3], shape[4], 1))
        
        DF = tf.multiply(DO, _AI)
        DF = tf.reduce_sum(DF, axis=0)
        DB = tf.reduce_sum(DO, axis=[0, 1, 2, 3, 4])

        self.filters = self.filters.assign(tf.subtract(self.filters, tf.scalar_mul(self.alpha, DF)))
        self.bias = self.bias.assign(tf.subtract(self.bias, tf.scalar_mul(self.alpha, DB)))
        return [(DF, self.filters), (DB, self.bias)]
    '''
    
    ###################################################################

    def dfa_backward(self, AI, AO, E, DO):
        pass
        
    def dfa_gv(self, AI, AO, E, DO):
        pass
        
    def dfa(self, AI, AO, E, DO): 
        pass
        
    ###################################################################    
        
        
        
        
