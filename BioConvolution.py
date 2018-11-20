
import tensorflow as tf
import numpy as np
import math

from Layer import Layer 
from Activation import Activation
from Activation import Sigmoid

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
    
class BioConvolution(Layer):

    def __init__(self, input_sizes, filter_sizes, num_classes, init_filters, strides, padding, alpha, activation: Activation, bias, last_layer, name=None, load=None, train=True):
        self.input_sizes = input_sizes
        self.filter_sizes = filter_sizes
        self.num_classes = num_classes
        
        # self.h and self.w only equal this for input sizes when padding = "SAME"...
        self.batch_size, self.h, self.w, self.fin = self.input_sizes
        self.fh, self.fw, self.f = self.filter_sizes

        self.bias = tf.Variable(tf.ones(shape=self.fout) * bias)

        self.strides = strides
        self.padding = padding

        self.alpha = alpha

        self.activation = activation
        self.last_layer = last_layer

        self.name = name
        self._train = train
        
        shape = (1, self.h*self.w, self.fh, self.fw, self.fin, self.fout)
        
        if init_filters == "zero":
            self.filters = tf.Variable(tf.zeros(shape=shape))
        elif init_filters == "sqrt_fan_in":
            sqrt_fan_in = math.sqrt(self.h*self.w*self.fin)
            self.filters = tf.Variable(tf.random_uniform(shape=shape, minval=-1.0/sqrt_fan_in, maxval=1.0/sqrt_fan_in))
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
        _X = image_to_patches(X, self.fw, self.fw)
        
        Z = tf.multiply(_X, self.filters)
        Z = tf.reduce_sum(Z, axis=(3, 4, 5))
        Z = Z + tf.reshape(self.bias, (1, 1, -1))
        
        A = self.activation.forward(Z)
        A = patches_to_image(A, self.h // self.fh, self.w // self.fw)
        return A
        
    ###################################################################           
        
    def backward(self, AI, AO, DO):
        # E
        DO = tf.multiply(DO, self.activation.gradient(AO))
        # dont think we can actually do this
        # DO here will be Nx10x10x64 ... cant make it a patch
        # DO = image_to_patches(DO, self.h // self.fh, self.w // self.fw)
        # but wait we have [Nx10x10x64]
        # and we want [N,100x64]
        # so we can do [N, 100, 1, 1, 1, 64] * [1, 100, 3, 3, 3, 64]
        # hmmm what do we do here ? 
        # for now just gonna manually rehsape it.
        
        shape = tf.shape(DO)
        N = shape[0]
        img_height = shape[1]
        img_width = shape[2]
        channels = shape[3]
        DO = tf.reshape(DO, (N, img_height * img_width, 1, 1, 1, channels))
        
        # F
        # self.filters = [1, 100, 3, 3, 3, 64]
        
        # this depends on stride ... and is not correct.
        DI = tf.multiply(DO, self.filters)
        DI = tf.reduce_sum(DI, axis=(6))
        
        return DI

    def gv(self, AI, AO, DO):  
        if not self._train:
            return []

        AI = tf.extract_image_patches(images=AI, ksizes=[1, self.fh, self.fw, 1], strides=self.strides, rates=[1, 1, 1, 1], padding=self.padding)
        shape = tf.shape(AI)
        AI = tf.reshape(AI, (shape[0], shape[1] * shape[2], shape[3], 1))
    
        DO = tf.multiply(DO, self.activation.gradient(AO))
        
        shape = tf.shape(DO)
        DO = tf.reshape(DO, (shape[0], shape[1], 1, shape[2])) # [N, 144, 1, 64]
        
        shape = tf.shape(AI)
        AI = tf.reshape(AI, (shape[0], shape[1], shape[2], 1)) # [N, 144, 27, 1]
        
        DF = tf.multiply(AI, DO)
        DB = tf.reduce_sum(DO, axis=[0, 1, 2])

        return [(DF, self.filters), (DB, self.bias)]
        
    def train(self, AI, AO, DO): 
        pass
        
    ###################################################################

    def dfa_backward(self, AI, AO, E, DO):
        pass
        
    def dfa_gv(self, AI, AO, E, DO):
        pass
        
    def dfa(self, AI, AO, E, DO): 
        pass
        
    ###################################################################    
        
        
        
        
