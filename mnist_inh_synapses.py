
import numpy as np
import argparse
import keras

#######################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=1e-2)
args = parser.parse_args()

LAYER1 = 784
LAYER2 = 400
LAYER3 = 10

TRAIN_EXAMPLES = 60000
TEST_EXAMPLES = 10000

#######################################

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def sigmoid(x):
    return 1. / (1. + np.exp(-x))
  
def dsigmoid(x):
    return x * (1. - x)

def relu(x):
    return x * (x > 0)
  
def drelu(x):
    # USE A NOT Z
    return 1.0 * (x > 0)

#######################################

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

y_train = keras.utils.to_categorical(y_train, 10)
x_train = x_train.reshape(TRAIN_EXAMPLES, 784)
x_train = x_train.astype('float32')
x_train = x_train / np.max(x_train)

y_test = keras.utils.to_categorical(y_test, 10)
x_test = x_test.reshape(TEST_EXAMPLES, 784)
x_test = x_test.astype('float32')
x_test = x_test / np.max(x_test)

#######################################

# high = 1. / np.sqrt((LAYER1 + LAYER2) / 2.)
high = 1. / np.sqrt(LAYER1)
weights1 = np.random.uniform(low=-high, high=high, size=(LAYER1, LAYER2))
bias1 = np.zeros(shape=LAYER2)
rate1 = 0.75
sign1 = np.random.choice([-1., 1.], size=(LAYER1, LAYER2), replace=True, p=[1.-rate1, rate1])

# high = 1. / np.sqrt(LAYER2)
high = 1. / np.sqrt(LAYER2)
weights2 = np.random.uniform(low=-high, high=high, size=(LAYER2, LAYER3))
bias2 = np.zeros(shape=LAYER3)
rate2 = 0.75
sign2 = np.random.choice([-1., 1.], size=(LAYER2, LAYER3), replace=True, p=[1.-rate2, rate2])

# high = 1. / np.sqrt(LAYER3)
high = 1. / np.sqrt(LAYER2)
b2 = np.random.uniform(low=-high, high=high, size=(LAYER3, LAYER2))

#######################################

for epoch in range(args.epochs):
    print ("epoch: %d/%d" % (epoch, args.epochs))
    
    correct = 0
    for ex in range(0, TRAIN_EXAMPLES, 50):
        start = ex 
        stop = ex + 50
    
        A1 = x_train[start:stop]
        Z2 = np.dot(A1, weights1 * sign1) + bias1
        A2 = relu(Z2)
        Z3 = np.dot(A2, weights2 * sign2) + bias2
        A3 = softmax(Z3)
        
        labels = y_train[start:stop]
        
        correct += np.sum(np.argmax(A3, axis=1) == np.argmax(labels, axis=1))
        
        # think we need to use sigmoid bc we not dividing batch size.
        # in mnist_fc we use tanh not relu.
        D3 = A3 - labels
        D2 = np.dot(D3, np.transpose(weights2 * sign2)) * drelu(A2)
        
        DW2 = np.dot(np.transpose(A2), D3) * sign2 # is this correct ? 
        DB2 = np.sum(D3, axis=0) 
        
        DW1 = np.dot(np.transpose(A1), D2) * sign1 # is this correct ? 
        DB1 = np.sum(D2, axis=0)
        
        weights2 = np.clip(weights2 - args.lr * DW2, 1e-6, 1e6)
        weights1 = np.clip(weights1 - args.lr * DW1, 1e-6, 1e6)
        
        bias2 = bias2 - args.lr * DB2
        bias1 = bias1 - args.lr * DB1
        
    train_acc = 1. * correct / TRAIN_EXAMPLES
        
    correct = 0
    for ex in range(0, TEST_EXAMPLES, 50):
        start = ex 
        stop = ex + 50
    
        A1 = x_test[start:stop]
        Z2 = np.dot(A1, weights1 * sign1) + bias1
        A2 = relu(Z2)
        Z3 = np.dot(A2, weights2 * sign2) + bias2
        A3 = softmax(Z3)
        
        labels = y_test[start:stop]
        
        correct += np.sum(np.argmax(A3, axis=1) == np.argmax(labels, axis=1))
            
    test_acc = 1. * correct / TEST_EXAMPLES
    
    print ("train: %f test: %f" % (train_acc, test_acc))
    
    
    
    
    
    
