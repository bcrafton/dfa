
import tensorflow as tf
import numpy as np
import math

from Layer import Layer 
from Activation import Activation
from Activation import Sigmoid

class SparseFC(Layer):

    def __init__(self, size : tuple, num_classes : int, init_weights : str, alpha : float, activation : Activation, bias : float, last_layer : bool, name=None, load=None, train=True, rate=1.):
        # input size
        self.size = size
        self.last_layer = last_layer
        self.input_size, self.output_size = size
        self.num_classes = num_classes
        self.rate = rate

        self.bias = tf.Variable(tf.ones(shape=[self.output_size]) * bias)

        self.alpha = alpha

        self.activation = activation

        self.name = name
        self._train = train
        
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
        # self.total_connects = tf.Print(self.total_connects, [self.total_connects], message="")

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
        
        
        
