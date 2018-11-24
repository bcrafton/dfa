
import tensorflow as tf
import numpy as np
import math

from Layer import Layer 
from Activation import Activation
from Activation import Sigmoid
from Activation import Linear

def image_to_patches(image, kernel_size, kernel_stride):
    shape = tf.shape(image)
    N = shape[0]
    height = shape[1]
    width = shape[2]
    channels = shape[3]
    kh, kw = kernel_size
    sh, sw = kernel_stride
    
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
                slic = tf.reshape(tf.slice(pad, begin=[0, ii, jj, 0], size=[N, height, width, channels]), (N, 1, height, width, channels))
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
    
class BioConvolution(Layer):

    def __init__(self, input_sizes, filter_sizes, num_classes, init_filters, strides, padding, alpha, activation: Activation, bias, last_layer, name=None, load=None, train=True):
        self.input_sizes = input_sizes
        self.filter_sizes = filter_sizes
        self.num_classes = num_classes
        
        # self.h and self.w only equal this for input sizes when padding = "SAME"...
        self.batch_size, self.h, self.w, self.fin = self.input_sizes
        self.fh, self.fw, self.fin, self.fout = self.filter_sizes

        self.bias = tf.Variable(tf.ones(shape=self.fout) * bias)

        self.strides = strides
        _, self.sh, self.sw, _ = self.strides
        self.padding = padding

        self.alpha = alpha

        self.activation = activation
        self.last_layer = last_layer

        self.name = name
        self._train = train
        
        assert(self.fh == self.fw)
        assert(self.sh == self.sw)
        assert(self.sh == 1 or self.sh == self.fh)
        
        shape = (1, self.h // self.sh * self.w // self.sw, self.fh, self.fw, self.fin, self.fout)
        
        if init_filters == "zero":
            self.filters = tf.Variable(tf.zeros(shape=shape))
        elif init_filters == "sqrt_fan_in":
            # sqrt_fan_in = math.sqrt(self.h*self.w*self.fin)
            # self.filters = tf.Variable(tf.random_uniform(shape=shape, minval=-1.0/sqrt_fan_in, maxval=1.0/sqrt_fan_in))
            self.filters = tf.Variable(np.ones(shape=(shape)), dtype=tf.float32)
        elif init_filters == "alexnet":
            self.filters = tf.Variable(np.random.normal(loc=0.0, scale=0.01, size=shape), dtype=tf.float32)
        else:
            self.filters = tf.get_variable(name=self.name, shape=shape)

    ###################################################################

    def get_weights(self):
        return [(self.name, self.filters), (self.name + "_bias", self.bias)]

    def num_params(self):
        filter_weights_size = self.h*self.w * self.fh*self.fw * self.fout
        bias_weights_size = self.fout
        return filter_weights_size + bias_weights_size
                
    def forward(self, X):
        _X = image_to_patches(X, (self.fh, self.fw), (self.sh, self.sw))
        shape = tf.shape(_X)
        _X = tf.reshape(_X, (shape[0], shape[1], shape[2], shape[3], shape[4], 1))
        
        Z = tf.multiply(_X, self.filters)
        Z = tf.reduce_sum(Z, axis=(2, 3, 4))
        Z = Z + tf.reshape(self.bias, (1, 1, self.fout))
        Z = tf.reshape(Z, (-1, self.h // self.sh, self.w // self.sw, self.fout))

        A = self.activation.forward(Z)
        return A
        
    ###################################################################           
        
    def backward(self, AI, AO, DO):
        # E
        # shape = (-1, self.h // self.fh * self.w // self.fw, 1, 1, 1, self.fout)
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DO = tf.reshape(DO, (-1, self.h // self.sh * self.w // self.sw, 1, 1, 1, self.fout))
        
        # F
        # shape = (1, self.h // self.fh * self.w // self.fw, self.fh, self.fw, self.fin, self.fout)
        
        # DI
        DI = tf.multiply(DO, self.filters)
        DI = tf.reduce_sum(DI, axis=(5))
        DI = tf.reshape(DI, (-1, self.h // self.sh, self.w // self.sw, self.fin))
        
        return DI

    def gv(self, AI, AO, DO):  
        if not self._train:
            return []

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DO = tf.reshape(DO, (-1, self.h // self.sh * self.w // self.sw, 1, 1, 1, self.fout))
        
        _AI = image_to_patches(AI, (self.fh, self.fw), (self.sh, self.sw))
        shape = tf.shape(_AI)
        _AI = tf.reshape(_AI, (shape[0], shape[1], shape[2], shape[3], shape[4], 1))
        
        DF = tf.multiply(DO, _AI)
        DF = tf.reduce_sum(DF, axis=0)
        DB = tf.reduce_sum(DO, axis=[0, 1, 2, 3, 4])

        return [(DF, self.filters), (DB, self.bias)]
        
    def train(self, AI, AO, DO): 
        if not self._train:
            return []

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DO = tf.reshape(DO, (-1, self.h // self.sh * self.w // self.sw, 1, 1, 1, self.fout))
        
        _AI = image_to_patches(AI, (self.fh, self.fw), (self.sh, self.sw))
        shape = tf.shape(_AI)
        _AI = tf.reshape(_AI, (shape[0], shape[1], shape[2], shape[3], shape[4], 1))
        
        DF = tf.multiply(DO, _AI)
        DF = tf.reduce_sum(DF, axis=0)
        DB = tf.reduce_sum(DO, axis=[0, 1, 2, 3, 4])

        self.filters = self.filters.assign(tf.subtract(self.filters, tf.scalar_mul(self.alpha, DF)))
        self.bias = self.bias.assign(tf.subtract(self.bias, tf.scalar_mul(self.alpha, DB)))
        return [(DF, self.filters), (DB, self.bias)]
        
    ###################################################################

    def dfa_backward(self, AI, AO, E, DO):
        pass
        
    def dfa_gv(self, AI, AO, E, DO):
        pass
        
    def dfa(self, AI, AO, E, DO): 
        pass
        
    ###################################################################    
        
        
        
        
