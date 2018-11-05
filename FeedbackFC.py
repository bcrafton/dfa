
import tensorflow as tf
import numpy as np
import math

from Layer import Layer 
from Activation import Activation
from Activation import Sigmoid

from FeedbackMatrix import FeedbackMatrix

np.set_printoptions(threshold=np.inf)

class FeedbackFC(Layer):
    num = 0
    def __init__(self, size : tuple, num_classes : int, sparse : int, rank : int, name=None, load=None):
        self.size = size
        self.num_classes = num_classes
        self.sparse = sparse
        self.rank = rank
        self.input_size, self.output_size = self.size
        self.name = name

        if load:
            weight_dict = np.load(load).item()
            self.B = tf.cast(tf.Variable(weight_dict[self.name]), tf.float32)
        else:
            b = FeedbackMatrix(size=(self.num_classes, self.output_size), sparse=self.sparse, rank=self.rank)
            self.B = tf.cast(tf.Variable(b), tf.float32) 
  
            '''
            print("rank:", np.linalg.matrix_rank(b))
            print("sparse: ", np.sum(b != 0, axis=0))
            D, V = np.linalg.eig( np.dot(b, b.T) )
            print(D)
            print(np.max(b.T), np.min(b.T))
            '''

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
        
        
        
        
        
