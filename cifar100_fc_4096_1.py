
import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--alpha', type=float, default=1e-2)
parser.add_argument('--l2', type=float, default=0.)
parser.add_argument('--decay', type=float, default=1.)
parser.add_argument('--eps', type=float, default=1.)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--act', type=str, default='tanh')
parser.add_argument('--bias', type=float, default=0.)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dfa', type=int, default=0)
parser.add_argument('--sparse', type=int, default=0)
parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--init', type=str, default="sqrt_fan_in")
parser.add_argument('--opt', type=str, default="adam")
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--name', type=str, default="cifar100_fc")
parser.add_argument('--load', type=str, default=None)
args = parser.parse_args()

if args.gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

##############################################

import time
import tensorflow as tf
import keras
import math
import numpy as np

from Model import Model

from Layer import Layer 
from ConvToFullyConnected import ConvToFullyConnected
from FullyConnected import FullyConnected
from Convolution import Convolution
from MaxPool import MaxPool
from Dropout import Dropout
from FeedbackFC import FeedbackFC
from FeedbackConv import FeedbackConv

from Activation import Activation
from Activation import Sigmoid
from Activation import Relu
from Activation import Tanh
from Activation import Softmax
from Activation import LeakyRelu
from Activation import Linear

##############################################

cifar100 = tf.keras.datasets.cifar100.load_data()

##############################################

var = 1. / 3072.
std = np.sqrt(var)
w1 = np.random.normal(loc=0.0, scale=std, size=[3072, 4096])

var = 1. / 4096.
std = np.sqrt(var)
w2 = np.random.normal(loc=0.0, scale=std, size=[4096, 4096])

var = 1. / 4096.
std = np.sqrt(var)
w3 = np.random.normal(loc=0.0, scale=std, size=[4096, 100])

b2 = w3
b1 = np.dot(w2, w3)

b2 = b2.T
b1 = b1.T

##############################################

EPOCHS = args.epochs
TRAIN_EXAMPLES = 50000
TEST_EXAMPLES = 10000
BATCH_SIZE = args.batch_size

if args.act == 'tanh':
    act = Tanh()
elif args.act == 'relu':
    act = Relu()
else:
    assert(False)

##############################################

tf.set_random_seed(0)
tf.reset_default_graph()

batch_size = tf.placeholder(tf.int32, shape=())
dropout_rate = tf.placeholder(tf.float32, shape=())
learning_rate = tf.placeholder(tf.float32, shape=())

Y = tf.placeholder(tf.float32, [None, 100])
X = tf.placeholder(tf.float32, [None, 3072])

l0 = FullyConnected(size=[3072, 4096], num_classes=100, init_weights=args.init, alpha=learning_rate, activation=act, bias=args.bias, last_layer=False, l2=args.l2, name="fc1", weights=w1)
l1 = Dropout(rate=dropout_rate)
l2 = FeedbackFC(size=[3072, 4096], num_classes=100, sparse=args.sparse, rank=args.rank, name="fc1_fb", weights=b1)

l3 = FullyConnected(size=[4096, 4096], num_classes=100, init_weights=args.init, alpha=learning_rate, activation=act, bias=args.bias, last_layer=False, l2=args.l2, name="fc2", weights=w2)
l4 = Dropout(rate=dropout_rate)
l5 = FeedbackFC(size=[4096, 4096], num_classes=100, sparse=args.sparse, rank=args.rank, name="fc2_fb", weights=b2)

l6 = FullyConnected(size=[4096, 100], num_classes=100, init_weights=args.init, alpha=learning_rate, activation=Linear(), bias=args.bias, last_layer=True, l2=args.l2, name="fc3", weights=w3)

model = Model(layers=[l0, l1, l2, l3, l4, l5, l6])

##############################################

predict = model.predict(X=X)
weights = model.get_weights()

if args.opt == "adam" or args.opt == "rms" or args.opt == "decay":
    if args.dfa:
        grads_and_vars = model.dfa_gvs(X=X, Y=Y)
    else:
        grads_and_vars = model.gvs(X=X, Y=Y)
        
    if args.opt == "adam":
        train = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=args.eps).apply_gradients(grads_and_vars=grads_and_vars)
    elif args.opt == "rms":
        train = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.99, epsilon=args.eps).apply_gradients(grads_and_vars=grads_and_vars)
    elif args.opt == "decay":
        train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).apply_gradients(grads_and_vars=grads_and_vars)
    else:
        assert(False)

else:
    if args.dfa:
        train = model.dfa(X=X, Y=Y)
    else:
        train = model.train(X=X, Y=Y)

correct = tf.equal(tf.argmax(predict,1), tf.argmax(Y,1))
total_correct = tf.reduce_sum(tf.cast(correct, tf.float32))

##############################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()

(x_train, y_train), (x_test, y_test) = cifar100

mean = np.mean(x_train, axis=(0, 1, 2), keepdims=True)
std = np.std(x_train, axis=(0, 1, 2), keepdims=True)

# x_train = x_train / 255.
x_train = (x_train - mean) / std
x_train = x_train.reshape(TRAIN_EXAMPLES, 3072)
y_train = keras.utils.to_categorical(y_train, 100)

# x_test = x_test / 255.
x_test = (x_test - mean) / std
x_test = x_test.reshape(TEST_EXAMPLES, 3072)
y_test = keras.utils.to_categorical(y_test, 100)
##############################################

filename = args.name + '.results'
f = open(filename, "w")
f.write(filename + "\n")
f.write("total params: " + str(model.num_params()) + "\n")
f.close()

##############################################

train_accs = []
test_accs = []

for ii in range(EPOCHS):
    if args.opt == 'decay' or args.opt == 'gd':
        decay = np.power(args.decay, ii)
        lr = args.alpha * decay
    else:
        lr = args.alpha
        
    print (ii)
    
    #############################
    
    _count = 0
    _total_correct = 0
    
    for jj in range(int(TRAIN_EXAMPLES / BATCH_SIZE)):
        xs = x_train[jj*BATCH_SIZE:(jj+1)*BATCH_SIZE]
        ys = y_train[jj*BATCH_SIZE:(jj+1)*BATCH_SIZE]
        _correct, _ = sess.run([total_correct, train], feed_dict={batch_size: BATCH_SIZE, dropout_rate: 0.5, learning_rate: lr, X: xs, Y: ys})
        
        _total_correct += _correct
        _count += BATCH_SIZE

    train_acc = 1.0 * _total_correct / _count
    train_accs.append(train_acc)

    #############################

    _count = 0
    _total_correct = 0

    for jj in range(int(TEST_EXAMPLES / BATCH_SIZE)):
        xs = x_test[jj*BATCH_SIZE:(jj+1)*BATCH_SIZE]
        ys = y_test[jj*BATCH_SIZE:(jj+1)*BATCH_SIZE]
        _correct = sess.run(total_correct, feed_dict={batch_size: BATCH_SIZE, dropout_rate: 0.0, learning_rate: 0.0, X: xs, Y: ys})
        
        _total_correct += _correct
        _count += BATCH_SIZE
        
    test_acc = 1.0 * _total_correct / _count
    test_accs.append(test_acc)
    
    #############################
            
    print ("train acc: %f test acc: %f" % (train_acc, test_acc))
    
    f = open(filename, "a")
    f.write(str(test_acc) + "\n")
    f.close()

    #############################

    [w] = sess.run([weights], feed_dict={})
    w['train_acc'] = train_accs
    w['test_acc'] = test_accs
    np.save(args.name, w)
    
    ##############################################

