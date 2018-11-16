
import tensorflow as tf
import numpy as np
import math

from Layer import Layer 
from Activation import Activation
from Activation import Sigmoid

def genGabor(sz, omega, theta, func=np.cos, K=np.pi):
    radius = (int(sz[0]/2.0), int(sz[1]/2.0))
    [x, y] = np.meshgrid(range(-radius[0], radius[0]+1), range(-radius[1], radius[1]+1))

    x1 = x * np.cos(theta) + y * np.sin(theta)
    y1 = -x * np.sin(theta) + y * np.cos(theta)
    
    gauss = omega**2 / (4*np.pi * K**2) * np.exp(- omega**2 / (8*K**2) * ( 4 * x1**2 + y1**2))
    sinusoid = func(omega * x1) * np.exp(K**2 / 2)
    gabor = gauss * sinusoid
    return gabor

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
        
        if init_filters == "zero":
            self.filters = tf.Variable(tf.zeros(shape=self.filter_sizes))
        elif init_filters == "sqrt_fan_in":
            sqrt_fan_in = math.sqrt(self.h*self.w*self.fin)
            self.filters = tf.Variable(tf.random_uniform(shape=self.filter_sizes, minval=-1.0/sqrt_fan_in, maxval=1.0/sqrt_fan_in))
        elif init_filters == "alexnet":
            # self.filters = tf.random_normal(shape=self.filter_sizes, mean=0.0, stddev=0.01)
            _filters = np.random.normal(loc=0.0, scale=0.01, size=self.filter_sizes)
            self.filters = tf.Variable(_filters, dtype=tf.float32)
        else:
            self.filters = tf.get_variable(name=self.name, shape=self.filter_sizes)

        if load:
            conv = np.load(load).item()[self.name]
            # self.fb = tf.Variable(conv, dtype=tf.float32)
            std = np.std(conv)
            avg = np.average(conv)
            
            # shape = (5, 5, 3, 96)
            shape = np.shape(conv)
            
            fb = np.zeros(shape=shape)
            
            n = shape[2]
            m = shape[3]
            for ii in range(n):
                for jj in range(m):
                    fb[:, :, ii, jj] = genGabor((5, 5), np.random.uniform(-1., 1.), np.random.uniform(-np.pi, np.pi), func=np.cos) 
                
            fb = fb * (avg / np.average(fb))
            self.fb = tf.Variable(fb, dtype=tf.float32)
            
        else:
            sqrt_fan_in = math.sqrt(self.h*self.w*self.fin)
            self.fb = tf.Variable(tf.random_uniform(shape=self.filter_sizes, minval=-1.0/sqrt_fan_in, maxval=1.0/sqrt_fan_in))

    ###################################################################

    def get_weights(self):
        return [(self.name, self.filters), (self.name + "_bias", self.bias)]

    def num_params(self):
        filter_weights_size = self.fh * self.fw * self.fin * self.fout
        bias_weights_size = self.fout
        return filter_weights_size + bias_weights_size
                
    def forward(self, X):
        Z = tf.add(tf.nn.conv2d(X, self.filters, self.strides, self.padding), tf.reshape(self.bias, [1, 1, self.fout]))
        # Z = tf.Print(Z, [tf.shape(Z)], message=self.name, summarize=25)
        A = self.activation.forward(Z)
        return A
        
    ###################################################################           
        
    def backward(self, AI, AO, DO):    
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DI = tf.nn.conv2d_backprop_input(input_sizes=self.input_sizes, filter=self.fb, out_backprop=DO, strides=self.strides, padding=self.padding)
        return DI

    def gv(self, AI, AO, DO):    
        if not self._train:
            return []
    
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DF = tf.nn.conv2d_backprop_filter(input=AI, filter_sizes=self.filter_sizes, out_backprop=DO, strides=self.strides, padding=self.padding)
        DB = tf.reduce_sum(DO, axis=[0, 1, 2])
        return [(DF, self.filters), (DB, self.bias)]
        
    def train(self, AI, AO, DO): 
        if not self._train:
            return []

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DF = tf.nn.conv2d_backprop_filter(input=AI, filter_sizes=self.filter_sizes, out_backprop=DO, strides=self.strides, padding=self.padding)
        DB = tf.reduce_sum(DO, axis=[0, 1, 2])

        # DF = tf.Print(DF, [tf.reduce_mean(DF), tf.keras.backend.std(DF), tf.reduce_mean(self.filters), tf.keras.backend.std(self.filters)], message="Conv: ")
        # DF = tf.Print(DF, [tf.shape(DF), tf.shape(self.filters)], message="", summarize=25)

        self.filters = self.filters.assign(tf.subtract(self.filters, tf.scalar_mul(self.alpha, DF)))
        self.bias = self.bias.assign(tf.subtract(self.bias, tf.scalar_mul(self.alpha, DB)))
        return [(DF, self.filters), (DB, self.bias)]
        
    ###################################################################

    def dfa_backward(self, AI, AO, E, DO):
        assert(False)
        return tf.ones(shape=(tf.shape(AI)))
        
    def dfa_gv(self, AI, AO, E, DO):
        assert(False)
        if not self._train:
            return []
    
        DO = tf.multiply(DO, self.activation.gradient(AO))
        DF = tf.nn.conv2d_backprop_filter(input=AI, filter_sizes=self.filter_sizes, out_backprop=DO, strides=self.strides, padding=self.padding)
        DB = tf.reduce_sum(DO, axis=[0, 1, 2])
        return [(DF, self.filters), (DB, self.bias)]
        
    def dfa(self, AI, AO, E, DO): 
        assert(False)
        if not self._train:
            return []

        DO = tf.multiply(DO, self.activation.gradient(AO))
        DF = tf.nn.conv2d_backprop_filter(input=AI, filter_sizes=self.filter_sizes, out_backprop=DO, strides=self.strides, padding=self.padding)
        DB = tf.reduce_sum(DO, axis=[0, 1, 2])

        # DF = tf.Print(DF, [tf.reduce_mean(DF), tf.keras.backend.std(DF), tf.reduce_mean(self.filters), tf.keras.backend.std(self.filters)], message="Conv: ")
        # DF = tf.Print(DF, [tf.shape(DF), tf.shape(self.filters)], message="", summarize=25)

        self.filters = self.filters.assign(tf.subtract(self.filters, tf.scalar_mul(self.alpha, DF)))
        self.bias = self.bias.assign(tf.subtract(self.bias, tf.scalar_mul(self.alpha, DB)))
        return [(DF, self.filters), (DB, self.bias)]
        
    ###################################################################    
        
        
        
        
