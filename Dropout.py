
import tensorflow as tf
import numpy as np
import math

from Layer import Layer 

class Dropout(Layer):

    def __init__(self, rate):
        self.rate = rate

    ###################################################################

    def get_weights(self):
        return []

    def num_params(self):
        return 0

    def forward(self, X, dropout=False):
        if dropout:
            self.dropout_mask = tf.cast(tf.random_uniform(shape=tf.shape(X)) > self.rate, tf.float32) # np.random.binomial(size=X.shape, n=1, p=1 - self.rate)
            return X * self.dropout_mask
        else:
            return X
            
    ###################################################################

    def backward(self, AI, AO, DO):
        return DO * self.dropout_mask
        
    def gv(self, AI, AO, DO):
        return []
        
    def train(self, AI, AO, DO): 
        return []
        
    ###################################################################

    def dfa(self, AI, AO, E, DO):
        return DO * self.dropout_mask

    def dfa_gv(self, AI, AO, E, DO):
        return []

    def dfa(self, AI, AO, E, DO):
        return []
