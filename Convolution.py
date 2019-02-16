
import tensorflow as tf
import numpy as np
import math

from Layer import Layer 
from Activation import Activation
from Activation import Sigmoid

class Convolution(Layer):

    def __init__(self, input_sizes, filter_sizes, num_classes, init_filters, strides, padding, alpha, activation: Activation, bias, last_layer, name=None, load=None, train=True):
        self.input_sizes = input_sizes
        self.filter_sizes = filter_sizes
        self.num_classes = num_classes
        
        # self.h and self.w only equal this for input sizes when padding = "SAME"...
        self.batch_size, self.h, self.w, self.fin = self.input_sizes
        self.fh, self.fw, self.fin, self.fout = self.filter_sizes

        self.bias = tf.Variable(tf.ones(shape=self.fout) * bias)

        self.strides = strides
        self.padding = padding

        self.alpha = alpha

        self.activation = activation
        self.last_layer = last_layer

        self.name = name
        self._train = train
        
        mask = np.random.choice([-1., 1.], size=self.filter_sizes, replace=True, p=[0.5, 0.5])
        # mask = np.random.choice([-1., 1.], size=(1, self.h, self.w, self.fin), replace=True, p=[0.25, 0.75])
        
        if load:
            assert(False)
        else:
            if init_filters == "zero":
                filters = np.ones(shape=self.filter_sizes) * 1e-6
            elif init_filters == "sqrt_fan_in":
                sqrt_fan_in = np.sqrt(self.h * self.w * self.fin)
                filters = np.random.uniform(low=1e-6, high=1.0/sqrt_fan_in, size=self.filter_sizes)
            elif init_filters == "alexnet":
                assert(False)
            else:
                assert(False)
                
        self.filters = tf.Variable(filters, dtype=tf.float32)
        self.mask = tf.Variable(mask, dtype=tf.float32)
                
    ###################################################################

    def get_weights(self):
        return [(self.name, self.filters), (self.name + "_bias", self.bias)]

    def num_params(self):
        filter_weights_size = self.fh * self.fw * self.fin * self.fout
        bias_weights_size = self.fout
        return filter_weights_size + bias_weights_size
                
    def forward(self, X):
        Z = tf.nn.conv2d(X, tf.clip_by_value(self.filters, 1e-6, 1e6) * self.mask, self.strides, self.padding) + tf.reshape(self.bias, [1, 1, self.fout])
        A = self.activation.forward(Z)
        return A
        
    ###################################################################           
        
    def backward(self, AI, AO, DO):    
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DI = tf.nn.conv2d_backprop_input(input_sizes=self.input_sizes, filter=tf.clip_by_value(self.filters, 1e-6, 1e6) * self.mask, out_backprop=DO, strides=self.strides, padding=self.padding)
        return DI

    def gv(self, AI, AO, DO):
        if not self._train:
            return []
    
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DF = tf.nn.conv2d_backprop_filter(input=AI, filter_sizes=self.filter_sizes, out_backprop=DO, strides=self.strides, padding=self.padding)
        DF = tf.multiply(DF, self.mask)
        DB = tf.reduce_sum(DO, axis=[0, 1, 2])
        
        return [(DF, self.filters), (DB, self.bias)]
        
    def train(self, AI, AO, DO): 
        if not self._train:
            return []

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DF = tf.nn.conv2d_backprop_filter(input=AI * self.mask, filter_sizes=self.filter_sizes, out_backprop=DO, strides=self.strides, padding=self.padding)
        # DF = tf.multiply(DF, self.mask)
        DB = tf.reduce_sum(DO, axis=[0, 1, 2])

        filters = tf.clip_by_value(self.filters - self.alpha * DF, 1e-6, 1e6)
        bias = self.bias - self.alpha * DB

        self.filters = self.filters.assign(filters)
        self.bias = self.bias.assign(bias)

        return [(DF, self.filters), (DB, self.bias)]
        
    ###################################################################

    def dfa_backward(self, AI, AO, E, DO):
        return tf.ones(shape=(tf.shape(AI)))
        
    def dfa_gv(self, AI, AO, E, DO):
        if not self._train:
            return []
    
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DF = tf.nn.conv2d_backprop_filter(input=AI * self.mask, filter_sizes=self.filter_sizes, out_backprop=DO, strides=self.strides, padding=self.padding)
        # DF = tf.multiply(DF, self.mask)
        DB = tf.reduce_sum(DO, axis=[0, 1, 2])
        
        return [(DF, self.filters), (DB, self.bias)]
        
    def dfa(self, AI, AO, E, DO): 
        if not self._train:
            return []

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DF = tf.nn.conv2d_backprop_filter(input=AI * self.mask, filter_sizes=self.filter_sizes, out_backprop=DO, strides=self.strides, padding=self.padding)
        # DF = tf.multiply(DF, self.mask)
        DB = tf.reduce_sum(DO, axis=[0, 1, 2])

        filters = tf.clip_by_value(self.filters - self.alpha * DF, 1e-6, 1e6)
        bias = self.bias - self.alpha * DB

        self.filters = self.filters.assign(filters)
        self.bias = self.bias.assign(bias)

        return [(DF, self.filters), (DB, self.bias)]
        
    ###################################################################    
        
    def lel_backward(self, AI, AO, E, DO, Y):
        return tf.ones(shape=(tf.shape(AI)))
        
    def lel_gv(self, AI, AO, E, DO, Y):
        if not self._train:
            return []
    
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DF = tf.nn.conv2d_backprop_filter(input=AI, filter_sizes=self.filter_sizes, out_backprop=DO, strides=self.strides, padding=self.padding)
        DB = tf.reduce_sum(DO, axis=[0, 1, 2])
        return [(DF, self.filters), (DB, self.bias)]
        
    def lel(self, AI, AO, E, DO, Y): 
        if not self._train:
            return []

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DF = tf.nn.conv2d_backprop_filter(input=AI, filter_sizes=self.filter_sizes, out_backprop=DO, strides=self.strides, padding=self.padding)
        DB = tf.reduce_sum(DO, axis=[0, 1, 2])

        self.filters = self.filters.assign(tf.subtract(self.filters, tf.scalar_mul(self.alpha, DF)))
        self.bias = self.bias.assign(tf.subtract(self.bias, tf.scalar_mul(self.alpha, DB)))
        return [(DF, self.filters), (DB, self.bias)]
        
    ################################################################### 
        
        
