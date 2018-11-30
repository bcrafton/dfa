
import numpy as np
import math
import gzip
import time
import pickle
import argparse
import keras
from keras.datasets import mnist

#######################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--alpha', type=float, default=1e-2)
parser.add_argument('--shuffle', type=int, default=1)
parser.add_argument('--num', type=int, default=0)
parser.add_argument('--load', type=int, default=0)
args = parser.parse_args()

#######################################

TRAIN_EXAMPLES = 60000
TEST_EXAMPLES = 10000
NUM_CLASSES = 10

LAYER1 = 784
LAYER2 = 100
LAYER3 = 10

#######################################

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(TRAIN_EXAMPLES, 784)
x_test = x_test.reshape(TEST_EXAMPLES, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = keras.utils.to_categorical(y_test, NUM_CLASSES)
    
#######################################

def sigmoid(x):
  return 1. / (1. + np.exp(-x))
  
def dsigmoid(x):
  # USE A NOT Z
  return x * (1. - x)

def tanh(x):
  return np.tanh(x)
  
def dtanh(x):
  # USE A NOT Z
  return (1. - (x ** 2))
  
def relu(x):
  return np.maximum(x, 0, x)
  
def drelu(x):
  # USE A NOT Z
  return x > 0
  
def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

#######################################

if args.load:
    weights1 = np.zeros(shape=(LAYER1, LAYER2))
    bias1 = np.zeros(shape=(LAYER2))
    
    weights2 = np.zeros(shape=(LAYER2, LAYER3))
    bias2 = np.zeros(shape=(LAYER3))

    B = np.load("W2_init.npy")
else:
    weights1 = np.zeros(shape=(LAYER1, LAYER2))
    bias1 = np.zeros(shape=(LAYER2))
    
    weights2 = np.zeros(shape=(LAYER2, LAYER3))
    bias2 = np.zeros(shape=(LAYER3))

    sqrt_fan_in = np.sqrt(LAYER2)
    high = 1.0 / sqrt_fan_in
    low = -high
    B = np.random.uniform(low=low, high=high, size=(LAYER2, LAYER3))

#######################################

for epoch in range(args.epochs):
    print ("epoch: " + str(epoch + 1) + "/" + str(args.epochs))

    for ex in range(0, TRAIN_EXAMPLES, args.batch_size): 

        if (TRAIN_EXAMPLES < ex + args.batch_size):
            num = ex + args.batch_size - TRAIN_EXAMPLES
        else:
            num = args.batch_size
            
        start = ex
        end = ex + num
            
        A1 = x_train[start:end]
        Z2 = np.dot(A1, weights1) + bias1
        A2 = tanh(Z2)
        Z3 = np.dot(A2, weights2) + bias2
        A3 = softmax(Z3)
        
        ANS = y_train[start:end]
                
        D3 = A3 - ANS
        S = np.dot(A2, B)
        Es = S - ANS
        D2 = np.dot(Es, B.T) * dtanh(A2)
        
        DW3 = np.dot(np.transpose(A2), D3)
        DB3 = np.sum(D3, axis=0)
        
        DW2 = np.dot(np.transpose(A1), D2)
        DB2 = np.sum(D2, axis=0)

        weights2 -= args.alpha * DW3    
        bias2 -= args.alpha * DB3    
        weights1 -= args.alpha * DW2
        bias1 -= args.alpha * DB2
        
    correct = 0
    for ex in range(0, TEST_EXAMPLES, args.batch_size):
        if (TEST_EXAMPLES < ex + args.batch_size):
            num = ex + args.batch_size - TEST_EXAMPLES
        else:
            num = args.batch_size
            
        start = ex
        end = ex + num
    
        A1 = x_test[start:end]
        Z2 = np.dot(A1, weights1) + bias1
        A2 = tanh(Z2)
        Z3 = np.dot(A2, weights2) + bias2
        A3 = softmax(Z3)
        
        correct += np.sum( np.argmax(A3, axis=1) == np.argmax(y_test[start:end], axis=1) )
        
    print ("accuracy: " + str(1.0 * correct / TEST_EXAMPLES))
    
np.save("W2_" + str(args.num), weights2)
np.save("W1_" + str(args.num), weights1)
    
    
    
