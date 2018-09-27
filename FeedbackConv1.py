
import tensorflow as tf
import numpy as np
import math

from Layer import Layer 
from Activation import Activation
from Activation import Sigmoid

def distance(scale0, p0, scale1, p1):
    x = (scale0 * (p0[0] - p1[0])) ** 2
    y = (scale1 * (p0[1] - p1[1])) ** 2
    return math.sqrt(x + y)

class FeedbackConv(Layer):
    num = 0
    def __init__(self, size : tuple, num_classes : int, sparse : bool, rank : int, name=None):
        self.size = size
        self.num_classes = num_classes
        self.sparse = sparse
        self.rank = rank
        self.batch_size, self.h, self.w, self.f = self.size
        self.name = name

        if self.rank and self.sparse:
            assert(self.rank >= self.sparse)

        #### CREATE THE SPARSE MASK ####
        if self.sparse:
            self.mask = np.zeros(shape=(self.f * self.h * self.w, self.num_classes))
            for ii in range(self.f * self.h * self.w):
                if self.rank > 0:
                    idx = np.random.randint(0, self.rank, size=self.sparse)
                else:
                    idx = np.random.randint(0, self.num_classes, size=self.sparse)
                self.mask[ii][idx] = 1.0
                
            self.mask = np.transpose(self.mask)
        else:
            self.mask = np.ones(shape=(self.num_classes, self.f * self.h * self.w))
        
        #### IF MATRIX HAS USER-SPECIFIED RANK ####
        sqrt_fan_out = np.sqrt(self.f * self.h * self.w)
        
        if self.rank > 0:
            lo = -1.0/np.sqrt(sqrt_fan_out)
            hi = 1.0/np.sqrt(sqrt_fan_out)
            
            b = np.zeros(shape=(self.f * self.h * self.w, self.num_classes))
            for ii in range(self.rank):
                tmp1 = np.random.uniform(lo, hi, size=(self.f * self.h * self.w, 1))
                tmp2 = np.random.uniform(lo, hi, size=(1, self.num_classes))
                b = b + (1.0 / self.rank) * np.dot(tmp1, tmp2)
                
            b = np.transpose(b)
            b = b * self.mask
            assert(np.linalg.matrix_rank(b) == self.rank)
            
            self.B = tf.cast(tf.Variable(b), tf.float32)
        else:
            # this looks broken af.
            hi = np.sqrt(6.0 / (self.num_classes + self.f * self.h + self.w))
            lo = -hi
        
            b = np.random.uniform(lo, hi, size=(self.num_classes, self.f * self.h * self.w))
            b = b * self.mask
            self.B = tf.cast(tf.Variable(b), tf.float32)
            

        ##########################################################

        print ("Making B")

        hi = 1.0 / sqrt_fan_out
        lo = -hi

        b = np.zeros(shape=(num_classes, self.f, self.h, self.w))
        for ii in range(num_classes):
            for jj in range(self.f):
                scale_x = np.random.uniform(low=0.1, high=1.0)
                scale_y = np.random.uniform(low=0.1, high=1.0)
                x = np.random.randint(low=0, high=self.h)
                y = np.random.randint(low=0, high=self.w)
                for kk in range(self.h):
                    for ll in range(self.w):
                        dist = distance(scale_x, (kk, ll), scale_y, (x, y))
                        b[ii][jj][kk][ll] = 1.0 / np.power(dist + 1., 0.25)
                        
                b[ii][jj] = b[ii][jj] - np.average(b[ii][jj])
                scale = hi / (np.average(np.absolute(b[ii][jj])))
                b[ii][jj] = b[ii][jj] * scale

        b = np.reshape(b, (num_classes, self.f * self.h * self.w))
        self.B = tf.cast(tf.Variable(b), tf.float32)

        ##########################################################
        
        FeedbackConv.num = FeedbackConv.num + 1

    def get_names(self):
        return [self.name]
    
    def get_weights(self):
        return [self.B]
    
    def get_feedback(self):
        return self.B

    def num_params(self):
        return 0
        
    def forward(self, X, dropout=False):
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
        E = tf.reshape(E, self.size)
        E = tf.multiply(E, DO)
        return E
        
    def dfa_gv(self, AI, AO, E, DO):
        return []
        
    def dfa(self, AI, AO, E, DO): 
        return []
        
    ###################################################################   
        
        
        
        
