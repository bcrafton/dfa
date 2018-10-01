
import numpy as np
import math
import gzip
import time
import pickle
import argparse

np.seterr(all='warn')

#######################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
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
    
def one_hot(value, max_value):
    # zero is first entry.
    assert(max_value == 9)
    
    ret = np.zeros(10)
    for ii in range(10):
        if value == ii:
            ret[ii] = 1
    
    return ret
    
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
  
# DO NOT USE THIS FOR BATCHES
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
 
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
    print ("epoch: " + str(epoch + 1) + "/" + str(args.epochs))
    
    for ex in range(TRAIN_EXAMPLES):
        A1 = training_set[ex]
        Z2 = np.dot(A1, weights1) + bias1
        A2 = relu(Z2)
        Z3 = np.dot(A2, weights2) + bias2
        A3 = softmax(Z3)
        
        ANS = one_hot(training_labels[ex], 9)
        
        D3 = A3 - ANS
        D2 = np.dot(D3, np.transpose(weights2)) * drelu(A2)
        
        DW3 = np.dot(A2.reshape(LAYER2, 1), D3.reshape(1, LAYER3))
        DB3 = D3
        DW2 = np.dot(A1.reshape(LAYER1, 1), D2.reshape(1, LAYER2))        
        DB2 = D2

        weights2 -= args.alpha * DW3    
        bias2 -= args.alpha * DB3    
        weights1 -= args.alpha * DW2
        bias1 -= args.alpha * DB2
        
    correct = 0
    for ex in range(TEST_EXAMPLES):
        A1 = testing_set[ex]
        Z2 = np.dot(A1, weights1) + bias1
        A2 = relu(Z2)
        Z3 = np.dot(A2, weights2) + bias2
        A3 = softmax(Z3)
        
        if (np.argmax(A3) == testing_labels[ex]):
            correct += 1
        
    print ("accuracy: " + str(1.0 * correct / TEST_EXAMPLES))
    
    
    
    
    
    
