
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
        
        if load:
            print ("Loading Weights: " + self.name)
            weight_dict = np.load(load).item()
            self.weights = tf.Variable(weight_dict[self.name])
            self.bias = tf.Variable(weight_dict[self.name + '_bias'])
        else:
            if init_weights == "zero":
                self.weights = tf.Variable(tf.zeros(shape=self.size))
            elif init_weights == "sqrt_fan_in":
                sqrt_fan_in = math.sqrt(self.input_size)
                self.weights = tf.Variable(tf.random_uniform(shape=self.size, minval=-1.0/sqrt_fan_in, maxval=1.0/sqrt_fan_in))
            elif init_weights == "alexnet":
                # self.weights = tf.random_normal(shape=self.size, mean=0.0, stddev=0.01)
                _weights = np.random.normal(loc=0.0, scale=0.01, size=self.size)
                self.weights = tf.Variable(_weights, dtype=tf.float32)
            else:
                self.weights = tf.get_variable(name=self.name, shape=self.size)

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
            
        N = tf.shape(AI)[0]
        N = tf.cast(N, dtype=tf.float32)
        
        DO = tf.multiply(DO, self.activation.gradient(AO))
        # sorta makes sense that we add regularization term even if weights are negative. want them to go back to zero
        # kinda weird its just weights ... not weights squared or something
        DW = tf.matmul(tf.transpose(AI), DO) + (self.l2 / N) * self.weights
        DB = tf.reduce_sum(DO, axis=0)

        # DW = tf.Print(DW, [tf.reduce_mean(DW), tf.keras.backend.std(DW), tf.reduce_mean(self.weights), tf.keras.backend.std(self.weights)], message=self.name)

        return [(DW, self.weights), (DB, self.bias)]

    def train(self, AI, AO, DO):
        if not self._train:
            return []

        N = tf.shape(AI)[0]
        N = tf.cast(N, dtype=tf.float32)

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DW = tf.matmul(tf.transpose(AI), DO) + (self.l2 / N) * self.weights
        DB = tf.reduce_sum(DO, axis=0)

        # DW = tf.Print(DW, [tf.reduce_mean(DW), tf.keras.backend.std(DW), tf.reduce_mean(self.weights), tf.keras.backend.std(self.weights)], message=self.name)

        self.weights = self.weights.assign(tf.subtract(self.weights, tf.scalar_mul(self.alpha, DW)))
        self.bias = self.bias.assign(tf.subtract(self.bias, tf.scalar_mul(self.alpha, DB)))
        return [(DW, self.weights), (DB, self.bias)]
        
    ###################################################################
    
    def dfa_backward(self, AI, AO, E, DO):
        return tf.ones(shape=(tf.shape(AI)))
        
    def dfa_gv(self, AI, AO, E, DO):
        if not self._train:
            return []

        N = tf.shape(AI)[0]
        N = tf.cast(N, dtype=tf.float32)

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DW = tf.matmul(tf.transpose(AI), DO) + (self.l2 / N) * self.weights
        DB = tf.reduce_sum(DO, axis=0)
        
        # DW = tf.Print(DW, [tf.reduce_mean(DW), tf.keras.backend.std(DW), tf.reduce_mean(self.weights), tf.keras.backend.std(self.weights)], message=self.name)
        
        return [(DW, self.weights), (DB, self.bias)]
        
    def dfa(self, AI, AO, E, DO):
        if not self._train:
            return []

        N = tf.shape(AI)[0]
        N = tf.cast(N, dtype=tf.float32)

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DW = tf.matmul(tf.transpose(AI), DO) + (self.l2 / N) * self.weights
        DB = tf.reduce_sum(DO, axis=0)

        # DW = tf.Print(DW, [tf.reduce_mean(DW), tf.keras.backend.std(DW), tf.reduce_mean(self.weights), tf.keras.backend.std(self.weights)], message=self.name)

        self.weights = self.weights.assign(tf.subtract(self.weights, tf.scalar_mul(self.alpha, DW)))
        self.bias = self.bias.assign(tf.subtract(self.bias, tf.scalar_mul(self.alpha, DB)))
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
        DW = tf.matmul(tf.transpose(AI), DO) + (self.l2 / N) * self.weights
        DB = tf.reduce_sum(DO, axis=0)
        
        # DW = tf.Print(DW, [tf.reduce_mean(DW), tf.keras.backend.std(DW), tf.reduce_mean(self.weights), tf.keras.backend.std(self.weights)], message=self.name)
        
        return [(DW, self.weights), (DB, self.bias)]
        
    def lel(self, AI, AO, E, DO, Y):
        if not self._train:
            return []

        N = tf.shape(AI)[0]
        N = tf.cast(N, dtype=tf.float32)

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DW = tf.matmul(tf.transpose(AI), DO) + (self.l2 / N) * self.weights
        DB = tf.reduce_sum(DO, axis=0)

        # DW = tf.Print(DW, [tf.reduce_mean(DW), tf.keras.backend.std(DW), tf.reduce_mean(self.weights), tf.keras.backend.std(self.weights)], message=self.name)

        self.weights = self.weights.assign(tf.subtract(self.weights, tf.scalar_mul(self.alpha, DW)))
        self.bias = self.bias.assign(tf.subtract(self.bias, tf.scalar_mul(self.alpha, DB)))
        
        return [(DW, self.weights), (DB, self.bias)]
        
        
        
