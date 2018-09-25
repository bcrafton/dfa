
import numpy as np
import math
import gzip
import time
import pickle
import argparse

#######################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=25)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--alpha', type=float, default=1e-2)
args = parser.parse_args()

#######################################

def load_data():
  global training_set, training_labels, testing_set, testing_labels
  f = gzip.open('mnist.pkl.gz', 'rb')
  train, valid, test = pickle.load(f) 
  f.close()

  [training_set, training_labels] = train
  [validation_set, validation_labels] = valid
  [testing_set, testing_labels] = test

  for i in range( len(training_set) ):
    training_set[i] = training_set[i].reshape(28*28)

  for i in range( len(testing_set) ):
    testing_set[i] = testing_set[i].reshape(28*28)
    
def one_hot(y, classes):
    assert(len(np.shape(y)) == 1)
    num = np.shape(y)[0]
    shape = (num, classes)
    
    ret = np.zeros(shape=shape)
    for ii in range(num):
        ret[ii][y[ii]] = 1.
    
    return np.array(ret)
    
#######################################
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
  
def sigmoid_gradient(x):
    gz = sigmoid(x)
    ret = gz * (1 - gz)
    return ret
  
def softmax(x):
    exp = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

#######################################
    
load_data()
np.random.seed(0)

#######################################

LAYER1 = 784
LAYER2 = 100
LAYER3 = 10

sqrt_fan_in = np.sqrt(LAYER1)
high = 1.0 / sqrt_fan_in
low = -high
weights1 = np.random.uniform(low=low, high=high, size=(LAYER1, LAYER2))
bias1 = np.zeros(shape=(LAYER2))

sqrt_fan_in = np.sqrt(LAYER2)
high = 1.0 / sqrt_fan_in
low = -high
weights2 = np.random.uniform(low=low, high=high, size=(LAYER2, LAYER3))
bias2 = np.zeros(shape=(LAYER3))

TRAIN_EXAMPLES = 50000
TEST_EXAMPLES = 10000
NUM_CLASSES = 10

#######################################

for epoch in range(args.epochs):
    print "epoch: " + str(epoch + 1) + "/" + str(args.epochs)

    for ex in range(0, TRAIN_EXAMPLES, args.batch_size):    
        if (TRAIN_EXAMPLES < ex + args.batch_size):
            num = ex + args.batch_size - TRAIN_EXAMPLES
        else:
            num = args.batch_size
            
        start = ex
        end = ex + num
            
        A1 = training_set[start:end]
        Z2 = np.dot(A1, weights1) + bias1
        A2 = sigmoid(Z2)
        Z3 = np.dot(A2, weights2) + bias2
        A3 = softmax(Z3)
        
        ANS = one_hot(training_labels[start:end], 10)
                
        D3 = (A3 - ANS)
        D2 = np.dot(D3, np.transpose(weights2)) * sigmoid_gradient(Z2)
                
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
    
        A1 = testing_set[start:end]
        Z2 = np.dot(A1, weights1) + bias1
        A2 = sigmoid(Z2)
        Z3 = np.dot(A2, weights2) + bias2
        A3 = Z3
        
        correct += np.sum(np.argmax(A3, axis=1) == testing_labels[start:end])
        
    print "accuracy: " + str(1.0 * correct / TEST_EXAMPLES)
    
    
    
    
    
    
