
import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--alpha', type=float, default=1e-4)
parser.add_argument('--eps', type=float, default=1e-5)
parser.add_argument('--l2', type=float, default=0.)
parser.add_argument('--decay', type=float, default=1.)
parser.add_argument('--rate', type=float, default=0.25)
parser.add_argument('--swap', type=float, default=0.0)
parser.add_argument('--sign', type=float, default=0.6)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--act', type=str, default='relu')
parser.add_argument('--bias', type=float, default=0.1)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dfa', type=int, default=0)
parser.add_argument('--sparse', type=int, default=0)
parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--init', type=str, default="sqrt_fan_in")
parser.add_argument('--opt', type=str, default="adam")
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--name', type=str, default="cifar10_conv")
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
from SparseFC import SparseFC
from SparseConv import SparseConv

from Activation import Activation
from Activation import Sigmoid
from Activation import Relu
from Activation import Tanh
from Activation import Softmax
from Activation import LeakyRelu
from Activation import Linear
from Activation import SignedRelu

##############################################

cifar10 = tf.keras.datasets.cifar10.load_data()

##############################################

EPOCHS = args.epochs
TRAIN_EXAMPLES = 50000
TEST_EXAMPLES = 10000
BATCH_SIZE = args.batch_size

train_fc=True
if args.load:
    train_conv=False
else:
    train_conv=True

weights_fc=None
weights_conv=args.load

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
swap = tf.placeholder(tf.bool, shape=())

X = tf.placeholder(tf.float32, [None, 32, 32, 3])
X = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), X)
Y = tf.placeholder(tf.float32, [None, 10])

l0 = SparseConv(input_sizes=[batch_size, 32, 32, 3], filter_sizes=[5, 5, 3, 96], num_classes=10, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, last_layer=False, name='conv1', load=weights_conv, train=train_conv, rate=args.rate, swap=args.swap, sign=args.sign)
l1 = MaxPool(size=[batch_size, 32, 32, 96], ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
l2 = FeedbackConv(size=[batch_size, 16, 16, 96], num_classes=10, sparse=args.sparse, rank=args.rank, name='conv1_fb')

l3 = SparseConv(input_sizes=[batch_size, 16, 16, 96], filter_sizes=[5, 5, 96, 128], num_classes=10, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, last_layer=False, name='conv2', load=weights_conv, train=train_conv, rate=args.rate, swap=args.swap, sign=args.sign)
l4 = MaxPool(size=[batch_size, 16, 16, 128], ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
l5 = FeedbackConv(size=[batch_size, 8, 8, 128], num_classes=10, sparse=args.sparse, rank=args.rank, name='conv2_fb')

l6 = SparseConv(input_sizes=[batch_size, 8, 8, 128], filter_sizes=[5, 5, 128, 256], num_classes=10, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, last_layer=False, name='conv3', load=weights_conv, train=train_conv, rate=args.rate, swap=args.swap, sign=args.sign)
l7 = MaxPool(size=[batch_size, 8, 8, 256], ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
l8 = FeedbackConv(size=[batch_size, 4, 4, 256], num_classes=10, sparse=args.sparse, rank=args.rank, name='conv3_fb')

l9 = ConvToFullyConnected(shape=[4, 4, 256])

l10 = SparseFC(size=[4*4*256, 2048], num_classes=10, init_weights=args.init, alpha=learning_rate, activation=Relu(), bias=args.bias, last_layer=False, name='fc1', load=weights_fc, train=train_fc, rate=args.rate, swap=args.swap, sign=args.sign)
l11 = Dropout(rate=dropout_rate)
l12 = FeedbackFC(size=[4*4*256, 2048], num_classes=10, sparse=args.sparse, rank=args.rank, name='fc1_fb')

l13 = SparseFC(size=[2048, 2048], num_classes=10, init_weights=args.init, alpha=learning_rate, activation=Relu(), bias=args.bias, last_layer=False, name='fc2', load=weights_fc, train=train_fc, rate=args.rate, swap=args.swap, sign=args.sign)
l14 = Dropout(rate=dropout_rate)
l15 = FeedbackFC(size=[2048, 2048], num_classes=10, sparse=args.sparse, rank=args.rank, name='fc2_fb')

l16 = SparseFC(size=[2048, 10], num_classes=10, init_weights=args.init, alpha=learning_rate, activation=Linear(), bias=args.bias, last_layer=True, name='fc3', load=weights_fc, train=train_fc, rate=1., swap=0., sign=1.)

##############################################

model = Model(layers=[l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15, l16])

SET = model.SET(swap)
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

# argmax axis = 1, want to argmax along the index, not batch
correct = tf.equal(tf.argmax(predict,1), tf.argmax(Y,1))
total_correct = tf.reduce_sum(tf.cast(correct, tf.float32))

# predictions: A Tensor of type float32. A batch_size x classes tensor.
# targets: A Tensor. Must be one of the following types: int32, int64. A batch_size vector of class ids.
# k: An int. Number of top elements to look at for computing precision.
top5 = tf.nn.in_top_k(predictions=predict, targets=tf.argmax(Y,1), k=5)
total_top5 = tf.reduce_sum(tf.cast(top5, tf.float32))

##############################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()

(x_train, y_train), (x_test, y_test) = cifar10

srv = True

if not srv:
  x_train = np.transpose(x_train, (0, 2, 3, 1))
  x_test = np.transpose(x_test, (0, 2, 3, 1))

assert(np.shape(x_train) == (TRAIN_EXAMPLES, 32, 32, 3))
assert(np.shape(x_test) == (TEST_EXAMPLES, 32, 32, 3))

# x_train = x_train.reshape(TRAIN_EXAMPLES, 32, 32, 3)
y_train = keras.utils.to_categorical(y_train, 10)

# x_test = x_test.reshape(TEST_EXAMPLES, 32, 32, 3)
y_test = keras.utils.to_categorical(y_test, 10)

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
        # print (jj)
    
        xs = x_train[jj*BATCH_SIZE:(jj+1)*BATCH_SIZE]
        ys = y_train[jj*BATCH_SIZE:(jj+1)*BATCH_SIZE]
        _correct, _gvs, _ = sess.run([total_correct, grads_and_vars, train], feed_dict={batch_size: BATCH_SIZE, dropout_rate: args.dropout, learning_rate: lr, X: xs, Y: ys})
        
        _total_correct += _correct
        _count += BATCH_SIZE

    train_acc = 1.0 * _total_correct / _count
    train_accs.append(train_acc)

    # print (np.shape(_gvs))
    for _gv in _gvs:
        tot = np.prod(np.shape(_gv[1]))
        neg = np.count_nonzero(_gv[1] < 0.)
        pos = np.count_nonzero(_gv[1] > 0.)
        print (np.shape(_gv[1]), 1.0*neg/tot, 1.0*pos/tot)

    #############################

    _count = 0
    _total_correct = 0
    
    for jj in range(int(TEST_EXAMPLES / BATCH_SIZE)):
        xs = x_test[jj*BATCH_SIZE:(jj+1)*BATCH_SIZE]
        ys = y_test[jj*BATCH_SIZE:(jj+1)*BATCH_SIZE]
        [_correct] = sess.run([total_correct], feed_dict={batch_size: BATCH_SIZE, dropout_rate: 0.0, learning_rate: 0.0, X: xs, Y: ys})
        
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
    
    _SET = sess.run(SET, feed_dict={batch_size: 0, dropout_rate: 0.0, learning_rate: 0.0, X: xs, Y: ys, swap: True})

    #############################

##############################################

if args.save:
    [w] = sess.run([weights], feed_dict={})
    w['train_acc'] = train_accs
    w['test_acc'] = test_accs
    np.save(args.name, w)
    
##############################################


