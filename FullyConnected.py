
import tensorflow as tf
import numpy as np
import math

from Layer import Layer 
from Activation import Activation
from Activation import Sigmoid

class FullyConnected(Layer):
    num = 0
    def __init__(self, size : tuple, num_classes : int, init_weights : str, alpha : float, activation : Activation, bias : float, last_layer : bool, name=None, load=None, train=True):
        
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

        # activation function
        self.activation = activation

        self.name = name
        self._train = train
        
        if load:
            weight_dict = np.load(load).item()
            self.weights = tf.Variable(weight_dict[self.name])
            self.bias = tf.Variable(weight_dict[self.name + '_bias'])
        else:
            if init_weights == "zero":
                self.weights = tf.Variable(tf.zeros(shape=self.size))
            elif init_weights == "sqrt_fan_in":
                sqrt_fan_in = math.sqrt(self.input_size)
                self.weights = tf.Variable(tf.random_uniform(shape=self.size, minval=-1.0/sqrt_fan_in, maxval=1.0/sqrt_fan_in))
            elif init_weights == "small":
                sqrt_fan_in = math.sqrt(self.input_size) / 10.
                self.weights = tf.Variable(tf.random_uniform(shape=self.size, minval=-1.0/sqrt_fan_in, maxval=1.0/sqrt_fan_in))
            else:
                self.weights = tf.get_variable(name="fc" + str(FullyConnected.num), shape=self.size)
                FullyConnected.num = FullyConnected.num + 1

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
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DW = tf.matmul(tf.transpose(AI), DO)
        DB = tf.reduce_sum(DO, axis=0)
        return [(DW, self.weights), (DB, self.bias)]

    def train(self, AI, AO, DO):
        if not self._train:
            return []

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DW = tf.matmul(tf.transpose(AI), DO)
        DB = tf.reduce_sum(DO, axis=0)

        # DW = tf.Print(DW, [tf.reduce_mean(DW), tf.keras.backend.std(DW), tf.reduce_mean(self.weights), tf.keras.backend.std(self.weights)], message=self.name)

        self.weights = self.weights.assign(tf.subtract(self.weights, tf.scalar_mul(self.alpha, DW)))
        self.bias = self.bias.assign(tf.subtract(self.bias, tf.scalar_mul(self.alpha, DB)))
        return [(DW, self.weights), (DB, self.bias)]
        
    ###################################################################
    
    def dfa_backward(self, AI, AO, E, DO):
        return tf.ones(shape=(tf.shape(AI)))
        
    def dfa_gv(self, AI, AO, E, DO):
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DW = tf.matmul(tf.transpose(AI), DO)
        DB = tf.reduce_sum(DO, axis=0)
        return [(DW, self.weights), (DB, self.bias)]
        
    def dfa(self, AI, AO, E, DO):
        if not self._train:
            return []

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DW = tf.matmul(tf.transpose(AI), DO)
        DB = tf.reduce_sum(DO, axis=0)

        # DW = tf.Print(DW, [tf.reduce_mean(DW), tf.keras.backend.std(DW), tf.reduce_mean(self.weights), tf.keras.backend.std(self.weights)], message=self.name)

        self.weights = self.weights.assign(tf.subtract(self.weights, tf.scalar_mul(self.alpha, DW)))
        self.bias = self.bias.assign(tf.subtract(self.bias, tf.scalar_mul(self.alpha, DB)))
        return [(DW, self.weights), (DB, self.bias)]
        
        
        
        
