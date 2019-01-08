
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

l0 = FullyConnected(size=[3072, 4096], num_classes=100, init_weights=args.init, alpha=learning_rate, activation=act, bias=args.bias, last_layer=False, l2=args.l2, name="fc1")
l1 = Dropout(rate=dropout_rate)
l2 = FeedbackFC(size=[3072, 4096], num_classes=100, sparse=args.sparse, rank=args.rank, name="fc1_fb")

l3 = FullyConnected(size=[4096, 4096], num_classes=100, init_weights=args.init, alpha=learning_rate, activation=act, bias=args.bias, last_layer=False, l2=args.l2, name="fc2")
l4 = Dropout(rate=dropout_rate)
l5 = FeedbackFC(size=[4096, 4096], num_classes=100, sparse=args.sparse, rank=args.rank, name="fc2_fb")

l6 = FullyConnected(size=[4096, 100], num_classes=100, init_weights=args.init, alpha=learning_rate, activation=Linear(), bias=args.bias, last_layer=True, l2=args.l2, name="fc3")

model = Model(layers=[l0, l1, l2, l3, l4, l5, l6])

##############################################

predict = model.predict(X=X)
A1 = model.up_to(X=X, N=0)
A2 = model.up_to(X=X, N=3)
A3 = model.up_to(X=X, N=6)

weights = model.get_weights()

dfa_gvs = model.dfa_gvs(X=X, Y=Y)
bp_gvs = model.gvs(X=X, Y=Y)

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

ratio1 = []
fc1 = []
dfc1 = []
dfc1_bias = []
a1 = []

ratio2 = []
fc2 = []
dfc2 = []
dfc2_bias = []
a2 = []

ratio3 = []
fc3 = []
dfc3 = []
dfc3_bias = []
a3 = []

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
        _correct, _gvs, _bp_gvs, _dfa_gvs, _A1, _A2, _A3, _ = sess.run([total_correct, grads_and_vars, bp_gvs, dfa_gvs, A1, A2, A3, train], feed_dict={batch_size: BATCH_SIZE, dropout_rate: 0.5, learning_rate: lr, X: xs, Y: ys})
        
        _total_correct += _correct
        _count += BATCH_SIZE
        
        ratio3.append(    np.sum(np.absolute(_dfa_gvs[0][0])) / np.sum(np.absolute(_bp_gvs[0][0])) )
        dfc3.append(      np.std(_gvs[0][0]) )
        dfc3_bias.append( np.std(_gvs[1][0]) )
        a3.append(        np.max(_A3)        )

        ratio2.append(    np.sum(np.absolute(_dfa_gvs[2][0])) / np.sum(np.absolute(_bp_gvs[2][0])) )
        dfc2.append(      np.std(_gvs[2][0]) )
        dfc2_bias.append( np.std(_gvs[3][0]) )
        a2.append(        np.max(_A2)        )

        ratio1.append(    np.sum(np.absolute(_dfa_gvs[4][0])) / np.sum(np.absolute(_bp_gvs[4][0])) )
        dfc1.append(      np.std(_gvs[4][0]) )
        dfc1_bias.append( np.std(_gvs[5][0]) )
        a1.append(        np.max(_A1)        )

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
    
    fc1.append( np.std(w['fc1']) )
    fc2.append( np.std(w['fc2']) )
    fc3.append( np.std(w['fc3']) )

    w = {}

    w['train_acc'] = train_accs
    w['test_acc'] = test_accs

    w['ratio1']    = ratio1
    w['fc1_std']   = fc1
    w['dfc1']      = dfc1
    w['dfc1_bias'] = dfc1_bias
    w['A1']        = a1

    w['ratio2']    = ratio2
    w['fc2_std']   = fc2
    w['dfc2']      = dfc2
    w['dfc2_bias'] = dfc2_bias
    w['A2']        = a2

    w['ratio3']    = ratio3
    w['fc3_std']   = fc3
    w['dfc3']      = dfc3 
    w['dfc3_bias'] = dfc3_bias
    w['A3']        = a3
    
    np.save(args.name, w)
    
    ##############################################

