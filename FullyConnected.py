
import tensorflow as tf
import numpy as np
import math

from Layer import Layer 
from Activation import Activation
from Activation import Sigmoid

class FullyConnected(Layer):

    def __init__(self, size, num_classes, init_weights, alpha, activation, bias, last_layer, l2=0., name=None, load=None, train=True):
        
        # TODO
        # check to make sure what we put in here is correct
        
        # input size
        self.size = size
        self.last_layer = last_layer
        self.input_size, self.output_size = size
        self.num_classes = num_classes

        # bias
        self.bias = tf.Variable(tf.ones(shape=[self.output_size]) * bias)

        # lr
        self.alpha = alpha

        # l2 loss lambda
        self.l2 = l2

        # activation function
        self.activation = activation

        self.name = name
        self._train = train
        
        mask = np.random.choice([-1., 1.], size=self.size, replace=True, p=[0.25, 0.75])
        # mask = np.random.choice([-1., 1.], size=(1, self.input_size), replace=True, p=[0.25, 0.75])
        # mask = np.repeat(mask, self.size[1], axis=1)
        # print (mask)
        
        if load:
            assert(False)
        else:
            if init_weights == "zero":
                weights = np.ones(shape=self.size) * 1e-6
            elif init_weights == "sqrt_fan_in":
                sqrt_fan_in = math.sqrt(self.input_size)
                weights = np.random.uniform(low=1e-6, high=1.0/sqrt_fan_in, size=self.size)
            elif init_weights == "alexnet":
                assert(False)
            else:
                assert(False)

        self.weights = tf.Variable(weights, dtype=tf.float32)
        self.mask = tf.Variable(mask, dtype=tf.float32)

    ###################################################################
        
    def get_weights(self):
        return [(self.name, self.weights), (self.name + "_bias", self.bias)]

    def num_params(self):
        weights_size = self.input_size * self.output_size
        bias_size = self.output_size
        return weights_size + bias_size

    def forward(self, X):
        Z = tf.matmul(X, tf.clip_by_value(self.weights, 1e-6, 1e6) * self.mask) + self.bias
        A = self.activation.forward(Z)
        return A

    ###################################################################
            
    def backward(self, AI, AO, DO):
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DI = tf.matmul(DO, tf.transpose(tf.clip_by_value(self.weights, 1e-6, 1e6) * self.mask))
        return DI
        
    def gv(self, AI, AO, DO):
        if not self._train:
                return []

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DW = tf.matmul(tf.transpose(AI), DO)
        DW = tf.multiply(DW, self.mask)
        DB = tf.reduce_sum(DO, axis=0)

        return [(DW, self.weights), (DB, self.bias)]

    def train(self, AI, AO, DO):
        if not self._train:
            return []

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DW = tf.matmul(tf.transpose(AI * self.mask), DO)
        # DW = tf.multiply(DW, self.mask)
        DB = tf.reduce_sum(DO, axis=0)

        weights = tf.clip_by_value(self.weights - self.alpha * DW, 1e-6, 1e6) 
        bias = self.bias - self.alpha * DB

        self.weights = self.weights.assign(weights)
        self.bias = self.bias.assign(bias)

        return [(DW, self.weights), (DB, self.bias)]
        
    ###################################################################
    
    def dfa_backward(self, AI, AO, E, DO):
        return tf.ones(shape=(tf.shape(AI)))
        
    def dfa_gv(self, AI, AO, E, DO):
        if not self._train:
            return []

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DW = tf.matmul(tf.transpose(AI * self.mask), DO)
        # DW = tf.multiply(DW, self.mask)
        DB = tf.reduce_sum(DO, axis=0)

        return [(DW, self.weights), (DB, self.bias)]
        
    def dfa(self, AI, AO, E, DO):
        if not self._train:
            return []
            
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DW = tf.matmul(tf.transpose(AI * self.mask), DO)
        # DW = tf.multiply(DW, self.mask)
        DB = tf.reduce_sum(DO, axis=0)

        weights = tf.clip_by_value(self.weights - self.alpha * DW, 1e-6, 1e6) 
        bias = self.bias - self.alpha * DB

        self.weights = self.weights.assign(weights)
        self.bias = self.bias.assign(bias)

        return [(DW, self.weights), (DB, self.bias)]
        
    ###################################################################
        
    def lel_backward(self, AI, AO, E, DO, Y):
        return tf.ones(shape=(tf.shape(AI)))
        
    def lel_gv(self, AI, AO, E, DO, Y):
        if not self._train:
            return []

        N = tf.shape(AI)[0]
        N = tf.cast(N, dtype=tf.float32)

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DW = tf.matmul(tf.transpose(AI), DO) + self.l2 * self.weights
        DB = tf.reduce_sum(DO, axis=0)

        return [(DW, self.weights), (DB, self.bias)]
        
    def lel(self, AI, AO, E, DO, Y):
        if not self._train:
            return []

        N = tf.shape(AI)[0]
        N = tf.cast(N, dtype=tf.float32)

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DW = tf.matmul(tf.transpose(AI), DO) + self.l2 * self.weights
        DB = tf.reduce_sum(DO, axis=0)

        self.weights = self.weights.assign(tf.subtract(self.weights, tf.scalar_mul(self.alpha, DW)))
        self.bias = self.bias.assign(tf.subtract(self.bias, tf.scalar_mul(self.alpha, DB)))
        
        return [(DW, self.weights), (DB, self.bias)]
        
        
        
