
import tensorflow as tf
import numpy as np
import math

from Layer import Layer 
from Activation import Activation
from Activation import Sigmoid

class FeedbackFC(Layer):
    num = 0
    def __init__(self, size : tuple, num_classes : int, sparse : int, rank : int, name=None, load=None):
        self.size = size
        self.num_classes = num_classes
        self.sparse = sparse
        self.rank = rank
        self.input_size, self.output_size = self.size
        self.name = name

        if self.rank and self.sparse:
            assert(self.rank >= self.sparse)

        if load:
            weight_dict = np.load(load).item()
            self.B = tf.cast(tf.Variable(weight_dict[self.name]), tf.float32)
        else:
            #### CREATE THE SPARSE MASK ####
            if self.sparse:
                self.mask = np.zeros(shape=(self.output_size, self.num_classes))
                for ii in range(self.output_size):
                    if self.rank > 0:
                        idx = np.random.randint(0, self.rank, size=self.sparse)
                    else:
                        idx = np.random.randint(0, self.num_classes, size=self.sparse)
                    self.mask[ii][idx] = 1.0
                    
                self.mask = np.transpose(self.mask)
            else:
                self.mask = np.ones(shape=(self.num_classes, self.output_size))
            
            #### IF MATRIX HAS USER-SPECIFIED RANK ####
            if sparse:
                sqrt_fan_out = np.sqrt(1.0 * self.output_size / self.num_classes * self.sparse)
            else:
                sqrt_fan_out = np.sqrt(self.output_size)

            if self.rank > 0:
                hi = 1.0 / sqrt_fan_out
                lo = -hi
                
                b = np.zeros(shape=(self.output_size, self.num_classes))
                for ii in range(self.rank):
                    tmp1 = np.random.uniform(lo, hi, size=(self.output_size, 1))
                    tmp2 = np.random.uniform(lo, hi, size=(1, self.num_classes))
                    b = b + np.dot(tmp1, tmp2)

                b = np.transpose(b)
                b = b * self.mask
                # this does take into account the sqrt(sparse)
                b = b * (hi / np.std(b))
                # print (np.std(b))
                assert(np.linalg.matrix_rank(b) == self.rank)
                
                self.B = tf.cast(tf.Variable(b), tf.float32)
            else:
                hi = 1.0 / sqrt_fan_out
                lo = -hi

                b = np.random.uniform(lo, hi, size=(self.num_classes, self.output_size))
                b = b * self.mask

                self.B = tf.cast(tf.Variable(b), tf.float32)            

            # self.B = tf.get_variable(name="feedback_fc_" + str(FeedbackFC.num), shape=(self.num_classes, self.output_size))
            # self.B = tf.Variable(tf.random_normal(mean=0.0, stddev=0.01, shape=(self.num_classes, self.output_size)))
            FeedbackFC.num = FeedbackFC.num + 1

    def get_weights(self):
        return [(self.name, self.B)]

    def get_feedback(self):
        return self.B

    def num_params(self):
        return 0
        
    def forward(self, X):
        return X
        
    ###################################################################           
        
    def backward(self, AI, AO, DO):    
        return DO

    def gv(self, AI, AO, DO):    
        return []
        
    def train(self, AI, AO, DO): 
        return []
        
    ###################################################################

    def dfa_backward(self, AI, AO, E, DO):
        E = tf.matmul(E, self.B)
        E = tf.multiply(E, DO)
        return E
        
    def dfa_gv(self, AI, AO, E, DO):
        return []
        
    def dfa(self, AI, AO, E, DO): 
        return []
        
    ###################################################################  
        
        
        
        
        
