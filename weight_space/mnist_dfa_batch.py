
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
parser.add_argument('--shuffle', type=int, default=1)
parser.add_argument('--num', type=int, default=0)
parser.add_argument('--load', type=int, default=0)
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

def tanh(x):
    return np.tanh(x)

def dtanh(x):
    # return 1 - np.power(x, 2)
    return 1. - x * x
   
def sigmoid(x):
    return 1. / (1. + np.exp(-x))
  
def dsigmoid(x):
    gz = sigmoid(x)
    ret = gz * (1 - gz)
    return ret

#######################################

LAYER1 = 784
LAYER2 = 100
LAYER3 = 10

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

    B = np.random.uniform(low=low, high=high, size=(LAYER2, LAYER3))

TRAIN_EXAMPLES = 50000
TEST_EXAMPLES = 10000
NUM_CLASSES = 10

#######################################

load_data()

if args.shuffle:
    print ("Shuffling!")
    perm = np.random.permutation(TRAIN_EXAMPLES)

    tmp1 = np.copy(training_set[0])
    training_set[perm] = training_set
    training_labels[perm] = training_labels
    tmp2 = training_set[perm[0]]
    
    assert(np.all(tmp1 == tmp2))

#######################################

for epoch in range(args.epochs):
    print "epoch: " + str(epoch + 1) + "/" + str(args.epochs)

    for ex in range(0, TRAIN_EXAMPLES, args.batch_size): 
  
        print (ex)

        if (TRAIN_EXAMPLES < ex + args.batch_size):
            num = ex + args.batch_size - TRAIN_EXAMPLES
        else:
            num = args.batch_size
            
        start = ex
        end = ex + num
            
        A1 = training_set[start:end]
        Z2 = np.dot(A1, weights1) + bias1
        A2 = tanh(Z2)
        Z3 = np.dot(A2, weights2) + bias2
        A3 = sigmoid(Z3)
        
        ANS = one_hot(training_labels[start:end], 10)
                
        D3 = (A3 - ANS)
        D2 = np.dot(D3, np.transpose(B)) * dtanh(Z2)
                
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
    
np.save("W2_" + str(args.num), weights2)
np.save("W1_" + str(args.num), weights1)
    
    
    
