
import tensorflow as tf
import numpy as np
import math

from Layer import Layer 
from Activation import Activation
from Activation import Sigmoid

class Sparse(Layer):

    def __init__(self, size : tuple, num_classes : int, init_weights : str, alpha : float, activation : Activation, bias : float, last_layer : bool, name=None, load=None, train=True, rate=1., swap=0.):
        self.size = size
        self.input_size, self.output_size = size
        self.num_classes = num_classes
        
        self.last_layer = last_layer
                
        self.rate = rate
        self.swap = swap
        self.nswap = int(self.rate * self.swap * self.input_size * self.output_size)

        self.bias = tf.Variable(tf.ones(shape=[self.output_size]) * bias)

        self.alpha = alpha

        self.activation = activation

        self.name = name
        self._train = train
        
        #########################################################

        _mask = np.zeros(self.size)
        for ii in range(self.input_size):
            idx = np.random.choice(range(self.output_size), size=int(self.rate * self.output_size), replace=False)
            _mask[ii][idx] = 1.
            
        _total_connects = int(np.count_nonzero(_mask))
        assert(_total_connects == int(self.rate * self.output_size) * self.input_size)
        
        assert(not load)
        if init_weights == "zero":
            _weights = np.zeros(shape=self.size)
        elif init_weights == "sqrt_fan_in":
            sqrt_fan_in = math.sqrt(self.input_size)
            _weights = np.random.uniform(low=-1.0/sqrt_fan_in, high=1.0/sqrt_fan_in, size=self.size)
        elif init_weights == "alexnet":
            _weights = np.random.normal(loc=0.0, scale=0.01, size=self.size)
        else:
            # Glorot
            assert(False)

        _weights = _mask * _weights
            
        self.weights = tf.Variable(_weights, dtype=tf.float32)
        self.mask = tf.Variable(_mask, dtype=tf.float32)
        self.total_connects = tf.Variable(tf.count_nonzero(self.mask))
        
        self._total_connects = np.count_nonzero(_mask)
        self.slice_size = self._total_connects - self.nswap
        
    ###################################################################
        
    def get_weights(self):
        return [(self.name, self.weights), (self.name + "_bias", self.bias)]

    def num_params(self):
        weights_size = self.input_size * self.output_size
        bias_size = self.output_size
        return weights_size + bias_size

    def forward(self, X):
        Z = tf.matmul(X, self.weights) + self.bias
        A = self.activation.forward(Z)
        return A

    ###################################################################
            
    def backward(self, AI, AO, DO):
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DI = tf.matmul(DO, tf.transpose(self.weights))
        return DI
        
    def gv(self, AI, AO, DO):
        if not self._train:
            return []

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DW = tf.matmul(tf.transpose(AI), DO)
        DB = tf.reduce_sum(DO, axis=0)
        return [(DW, self.weights), (DB, self.bias)]

    def train(self, AI, AO, DO):
        # assert(tf.count_nonzero(self.weights) == self.total_connects)
        # _assert = tf.assert_greater_equal(self.total_connects, tf.count_nonzero(self.weights))
        
        # with tf.control_dependencies([_assert]):
        if not self._train:
            return []

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DW = tf.matmul(tf.transpose(AI), DO)
        
        DB = tf.reduce_sum(DO, axis=0)

        self.weights = self.weights.assign(tf.subtract(self.weights, tf.scalar_mul(self.alpha, DW)))
        self.bias = self.bias.assign(tf.subtract(self.bias, tf.scalar_mul(self.alpha, DB)))
        return [(DW, self.weights), (DB, self.bias)]
        
    ###################################################################
    
    def dfa_backward(self, AI, AO, E, DO):
        return tf.ones(shape=(tf.shape(AI)))
        
    def dfa_gv(self, AI, AO, E, DO):
        if not self._train:
            return []

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DW = tf.matmul(tf.transpose(AI), DO)
        DB = tf.reduce_sum(DO, axis=0)
        return [(DW, self.weights), (DB, self.bias)]
        
    def dfa(self, AI, AO, E, DO):
        # assert(tf.count_nonzero(self.weights) == self.total_connects)
        _assert = tf.assert_greater_equal(self.total_connects, tf.count_nonzero(self.weights))
        
        with tf.control_dependencies([_assert]):
            if not self._train:
                return []

            DO = tf.multiply(DO, self.activation.gradient(AO))
            DW = tf.matmul(tf.transpose(AI), DO)
            DW = tf.multiply(DW, self.mask)
            DB = tf.reduce_sum(DO, axis=0)

            self.weights = self.weights.assign(tf.subtract(self.weights, tf.scalar_mul(self.alpha, DW)))
            self.bias = self.bias.assign(tf.subtract(self.bias, tf.scalar_mul(self.alpha, DB)))
            return [(DW, self.weights), (DB, self.bias)]
        
    ###################################################################
        
    def lel_backward(self, AI, AO, E, DO, Y):
        return tf.ones(shape=(tf.shape(AI)))
        
    def lel_gv(self, AI, AO, E, DO, Y):
        if not self._train:
            return []

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DW = tf.matmul(tf.transpose(AI), DO)
        DB = tf.reduce_sum(DO, axis=0)
        return [(DW, self.weights), (DB, self.bias)]
        
    def lel(self, AI, AO, E, DO, Y):
        if not self._train:
            return []

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DW = tf.matmul(tf.transpose(AI), DO)
        DB = tf.reduce_sum(DO, axis=0)

        self.weights = self.weights.assign(tf.subtract(self.weights, tf.scalar_mul(self.alpha, DW)))
        self.bias = self.bias.assign(tf.subtract(self.bias, tf.scalar_mul(self.alpha, DB)))
        return [(DW, self.weights), (DB, self.bias)]
        
    ###################################################################
    
    def SET(self):
        shape = tf.shape(self.weights)

        abs_w = tf.abs(self.weights)

        vld_i = tf.where(abs_w > 0)
        vld_w = tf.gather_nd(abs_w, vld_i)

        sorted_i = tf.contrib.framework.argsort(vld_w, axis=0)
        small_i = tf.gather(vld_i, sorted_i, axis=0)
        small_i = tf.cast(small_i, tf.int32)
        small_i = tf.slice(small_i, [0, 0], [self.nswap, 2])
        small_w = tf.zeros(shape=(self.nswap,))
        
        # prove that this code only runs 3 times.
        # because assertions fail when tensorflow builds the graph
        # sorted_i = tf.Print(sorted_i, [sorted_i], message="")
        
        new_i = tf.where(abs_w <= 0)
        new_i = tf.random_shuffle(new_i)
        new_i = tf.slice(new_i, [0, 0], [self.nswap, 2])
        new_i = tf.cast(new_i, tf.int32)
        sqrt_fan_in = math.sqrt(self.input_size)
        new_w = tf.random_uniform(minval=-1.0/sqrt_fan_in, maxval=1.0/sqrt_fan_in, shape=(self.nswap,))
        
        # vld_i = tf.where(abs_w > 0)
        # vld_w = tf.gather_nd(abs_w, vld_i)
        # sorted_i = tf.contrib.framework.argsort(vld_w, axis=0)
        large_i = tf.gather(vld_i, sorted_i, axis=0)
        large_i = tf.cast(large_i, tf.int32)
        large_i = tf.slice(large_i, [self.nswap, 0], [self.slice_size, 2])
        large_w = tf.gather_nd(self.weights, large_i)
        
        # dont need to assign the zeros here.
        # indices = tf.concat((large_i, small_i, new_i), axis=1)
        # updates = tf.concat((large_w, small_w, new_w), axis=0)
        indices = tf.concat((large_i, new_i), axis=0)
        updates = tf.concat((large_w, new_w), axis=0)
        weights = tf.scatter_nd(indices=indices, updates=updates, shape=shape)
        
        large_w = tf.ones(shape=(self.slice_size,))
        new_w = tf.ones(shape=(self.nswap,))
        updates = tf.concat((large_w, new_w), axis=0)
        mask = tf.scatter_nd(indices=indices, updates=updates, shape=shape)

        return [(mask, weights)]
        
    def NSET(self):    
        return [(self.mask, self.weights)]
        
        
        
