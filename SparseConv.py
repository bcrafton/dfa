
import tensorflow as tf
import numpy as np
import math

from Layer import Layer 
from Activation import Activation
from Activation import Sigmoid

class SparseConv(Layer):

    def __init__(self, input_sizes, filter_sizes, num_classes, init_filters, strides, padding, alpha, activation, bias, last_layer, name=None, load=None, train=True, rate=1., swap=0.):
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
        
        self.rate = rate
        self.swap = swap
        self.nswap = int(self.rate * self.swap * np.prod(self.filter_sizes))
        
        mask = np.random.choice([0, 1], size=self.filter_sizes, replace=True, p=[1.-rate, rate])
        
        # _total_connects = int(np.count_nonzero(mask))
        # print (_total_connects)
        # assert(_total_connects == int(self.rate * np.prod(self.filter_sizes)))
        
        if load:
            assert(False)
        else:
            if init_filters == "zero":
                filters = np.zeros(shape=self.filter_sizes)
            elif init_filters == "sqrt_fan_in":
                sqrt_fan_in = np.sqrt(self.h * self.w * self.fin)
                filters = np.random.uniform(low=-1.0/sqrt_fan_in, high=1.0/sqrt_fan_in, size=self.filter_sizes)
            elif init_filters == "alexnet":
                filters = np.random.normal(loc=0.0, scale=0.01, size=self.filter_sizes)
            else:
                # Glorot
                assert(False)
                
        filters = mask * filters

        self.filters = tf.Variable(filters, dtype=tf.float32)
        self.mask = tf.Variable(mask, dtype=tf.float32)
        self.total_connects = tf.Variable(tf.count_nonzero(self.mask))
        
        self._total_connects = np.count_nonzero(mask)
        self.slice_size = self._total_connects - self.nswap

    ###################################################################

    def get_weights(self):
        return [(self.name, self.filters), (self.name + "_bias", self.bias)]

    def num_params(self):
        filter_weights_size = np.prod(self.filter_sizes)
        bias_weights_size = self.fout
        return filter_weights_size + bias_weights_size
                
    def forward(self, X):
        Z = tf.add(tf.nn.conv2d(X, self.filters, self.strides, self.padding), tf.reshape(self.bias, [1, 1, self.fout]))
        A = self.activation.forward(Z)
        return A
        
    ###################################################################           
        
    def backward(self, AI, AO, DO):    
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DI = tf.nn.conv2d_backprop_input(input_sizes=self.input_sizes, filter=self.filters, out_backprop=DO, strides=self.strides, padding=self.padding)
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
        DF = tf.nn.conv2d_backprop_filter(input=AI, filter_sizes=self.filter_sizes, out_backprop=DO, strides=self.strides, padding=self.padding)
        DF = tf.multiply(DF, self.mask)
        DB = tf.reduce_sum(DO, axis=[0, 1, 2])

        # DF = tf.Print(DF, [tf.reduce_mean(DF), tf.keras.backend.std(DF), tf.reduce_mean(self.filters), tf.keras.backend.std(self.filters)], message="Conv: ")
        # DF = tf.Print(DF, [tf.shape(DF), tf.shape(self.filters)], message="", summarize=25)

        self.filters = self.filters.assign(tf.subtract(self.filters, tf.scalar_mul(self.alpha, DF)))
        self.bias = self.bias.assign(tf.subtract(self.bias, tf.scalar_mul(self.alpha, DB)))
        return [(DF, self.filters), (DB, self.bias)]
        
    ###################################################################

    def dfa_backward(self, AI, AO, E, DO):
        return tf.ones(shape=(tf.shape(AI)))
        
    def dfa_gv(self, AI, AO, E, DO):
        if not self._train:
            return []
    
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DF = tf.nn.conv2d_backprop_filter(input=AI, filter_sizes=self.filter_sizes, out_backprop=DO, strides=self.strides, padding=self.padding)
        DF = tf.multiply(DF, self.mask)
        DB = tf.reduce_sum(DO, axis=[0, 1, 2])
        return [(DF, self.filters), (DB, self.bias)]
        
    def dfa(self, AI, AO, E, DO): 
        if not self._train:
            return []

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DF = tf.nn.conv2d_backprop_filter(input=AI, filter_sizes=self.filter_sizes, out_backprop=DO, strides=self.strides, padding=self.padding)
        DF = tf.multiply(DF, self.mask)
        DB = tf.reduce_sum(DO, axis=[0, 1, 2])

        # DF = tf.Print(DF, [tf.reduce_mean(DF), tf.keras.backend.std(DF), tf.reduce_mean(self.filters), tf.keras.backend.std(self.filters)], message="Conv: ")
        # DF = tf.Print(DF, [tf.shape(DF), tf.shape(self.filters)], message="", summarize=25)

        self.filters = self.filters.assign(tf.subtract(self.filters, tf.scalar_mul(self.alpha, DF)))
        self.bias = self.bias.assign(tf.subtract(self.bias, tf.scalar_mul(self.alpha, DB)))
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
        
    def SET(self):
        shape = tf.shape(self.filters)
        abs_w = tf.abs(self.filters)
        vld_i = tf.where(abs_w > 0)
        vld_w = tf.gather_nd(abs_w, vld_i)
        sorted_i = tf.contrib.framework.argsort(vld_w, axis=0)
        
        # new indices
        new_i = tf.where(abs_w <= 0)
        new_i = tf.random_shuffle(new_i)
        new_i = tf.slice(new_i, [0, 0], [self.nswap, 4])
        new_i = tf.cast(new_i, tf.int32)
        sqrt_fan_in = np.sqrt(self.h * self.w * self.fin)
        new_w = tf.random_uniform(minval=-1.0/sqrt_fan_in, maxval=1.0/sqrt_fan_in, shape=(self.nswap,))
        
        # largest indices (rate - rate * nswap)
        large_i = tf.gather(vld_i, sorted_i, axis=0)
        large_i = tf.cast(large_i, tf.int32)
        large_i = tf.slice(large_i, [self.nswap, 0], [self.slice_size, 4])
        large_w = tf.gather_nd(self.filters, large_i)
        
        # update filters
        indices = tf.concat((large_i, new_i), axis=0)
        updates = tf.concat((large_w, new_w), axis=0)
        filters = tf.scatter_nd(indices=indices, updates=updates, shape=shape)
        
        # update mask
        large_w = tf.ones(shape=(self.slice_size,))
        new_w = tf.ones(shape=(self.nswap,))
        updates = tf.concat((large_w, new_w), axis=0)
        mask = tf.scatter_nd(indices=indices, updates=updates, shape=shape)

        # assign 
        self.filters = self.filters.assign(filters)
        self.mask = self.mask.assign(mask)

        return [(self.mask, self.filters)]
        
    def NSET(self):    
        return [(self.mask, self.filters)]
        
        
