
import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--alpha', type=float, default=1e-2)
parser.add_argument('--eps', type=float, default=1.)
parser.add_argument('--decay', type=float, default=1.)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--alg', type=str, default='bp')
parser.add_argument('--sparse', type=int, default=0)
parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--init', type=str, default="sqrt_fan_in")
parser.add_argument('--opt', type=str, default="adam")
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--name', type=str, default="cifar10_conv_weights")
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
from tensorflow.examples.tutorials.mnist import input_data

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

bias = 0.0

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

l0 = SparseConv(input_sizes=[batch_size, 32, 32, 3], filter_sizes=[5, 5, 3, 96], num_classes=10, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Tanh(), bias=bias, last_layer=False, name='conv1', load=weights_conv, train=train_conv, rate=0.25, swap=0.2)
l1 = MaxPool(size=[batch_size, 32, 32, 96], ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
l2 = FeedbackConv(size=[batch_size, 16, 16, 96], num_classes=10, sparse=args.sparse, rank=args.rank, name='conv1_fb')

l3 = SparseConv(input_sizes=[batch_size, 16, 16, 96], filter_sizes=[5, 5, 96, 128], num_classes=10, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Tanh(), bias=bias, last_layer=False, name='conv2', load=weights_conv, train=train_conv, rate=0.25, swap=0.2)
l4 = MaxPool(size=[batch_size, 16, 16, 128], ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
l5 = FeedbackConv(size=[batch_size, 8, 8, 128], num_classes=10, sparse=args.sparse, rank=args.rank, name='conv2_fb')

l6 = SparseConv(input_sizes=[batch_size, 8, 8, 128], filter_sizes=[5, 5, 128, 256], num_classes=10, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Tanh(), bias=bias, last_layer=False, name='conv3', load=weights_conv, train=train_conv, rate=0.25, swap=0.2)
l7 = MaxPool(size=[batch_size, 8, 8, 256], ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="SAME")
l8 = FeedbackConv(size=[batch_size, 4, 4, 256], num_classes=10, sparse=args.sparse, rank=args.rank, name='conv3_fb')

l9 = ConvToFullyConnected(shape=[4, 4, 256])

l10 = SparseFC(size=[4*4*256, 2048], num_classes=10, init_weights=args.init, alpha=learning_rate, activation=Tanh(), bias=bias, last_layer=False, name='fc1', load=weights_fc, train=train_fc, rate=0.25, swap=0.2)
l11 = Dropout(rate=dropout_rate)
l12 = FeedbackFC(size=[4*4*256, 2048], num_classes=10, sparse=args.sparse, rank=args.rank, name='fc1_fb')

l13 = SparseFC(size=[2048, 2048], num_classes=10, init_weights=args.init, alpha=learning_rate, activation=Tanh(), bias=bias, last_layer=False, name='fc2', load=weights_fc, train=train_fc, rate=0.25, swap=0.2)
l14 = Dropout(rate=dropout_rate)
l15 = FeedbackFC(size=[2048, 2048], num_classes=10, sparse=args.sparse, rank=args.rank, name='fc2_fb')

l16 = SparseFC(size=[2048, 10], num_classes=10, init_weights=args.init, alpha=learning_rate, activation=Linear(), bias=bias, last_layer=True, name='fc3', load=weights_fc, train=train_fc, rate=0.25, swap=0.2)

##############################################

model = Model(layers=[l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15, l16])

SET = model.SET(swap)
predict = model.predict(X=X)
weights = model.get_weights()

if args.opt == "adam" or args.opt == "rms" or args.opt == "decay":
    if args.alg == 'dfa':
        grads_and_vars = model.dfa_gvs(X=X, Y=Y)
    elif args.alg == 'lel':
        grads_and_vars = model.lel_gvs(X=X, Y=Y)
    elif args.alg == 'bp':
        grads_and_vars = model.gvs(X=X, Y=Y)
    else:
        assert(False)
        
    if args.opt == "adam":
        train = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=args.eps).apply_gradients(grads_and_vars=grads_and_vars)
    elif args.opt == "rms":
        train = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.99, epsilon=args.eps).apply_gradients(grads_and_vars=grads_and_vars)
    elif args.opt == "decay":
        train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).apply_gradients(grads_and_vars=grads_and_vars)
    else:
        assert(False)

else:
    if args.alg == 'dfa':
        train = model.dfa(X=X, Y=Y)
    elif args.alg == 'lel':
        train = model.lel(X=X, Y=Y)
    elif args.alg == 'bp':
        train = model.train(X=X, Y=Y)
    else:
        assert(False)

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

# assert(np.shape(x_train) == (TRAIN_EXAMPLES, 32, 32, 3))
# assert(np.shape(x_test) == (TEST_EXAMPLES, 32, 32, 3))

x_train = np.transpose(x_train, (0, 2, 3, 1))
x_test = np.transpose(x_test, (0, 2, 3, 1))

x_train = x_train.reshape(TRAIN_EXAMPLES, 32, 32, 3)
y_train = keras.utils.to_categorical(y_train, 10)

x_test = x_test.reshape(TEST_EXAMPLES, 32, 32, 3)
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

train_accs_top5 = []
test_accs_top5 = []

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
    _total_top5 = 0
    
    for jj in range(int(TRAIN_EXAMPLES / BATCH_SIZE)):
        xs = x_train[jj*BATCH_SIZE:(jj+1)*BATCH_SIZE]
        ys = y_train[jj*BATCH_SIZE:(jj+1)*BATCH_SIZE]
        _correct, _top5, _ = sess.run([total_correct, total_top5, train], feed_dict={batch_size: BATCH_SIZE, dropout_rate: 0.5, learning_rate: lr, X: xs, Y: ys})
        
        _total_correct += _correct
        _total_top5 += _top5
        _count += BATCH_SIZE

    train_acc = 1.0 * _total_correct / _count
    train_accs.append(train_acc)

    train_acc_top5 = 1.0 * _total_top5 / _count
    train_accs_top5.append(train_acc_top5)

    #############################

    _count = 0
    _total_correct = 0
    _total_top5 = 0
    
    for jj in range(int(TEST_EXAMPLES / BATCH_SIZE)):
        xs = x_test[jj*BATCH_SIZE:(jj+1)*BATCH_SIZE]
        ys = y_test[jj*BATCH_SIZE:(jj+1)*BATCH_SIZE]
        _correct, _top5 = sess.run([total_correct, total_top5], feed_dict={batch_size: BATCH_SIZE, dropout_rate: 0.0, learning_rate: 0.0, X: xs, Y: ys})
        
        _total_correct += _correct
        _total_top5 += _top5
        _count += BATCH_SIZE
        
    test_acc = 1.0 * _total_correct / _count
    test_accs.append(test_acc)

    test_acc_top5 = 1.0 * _total_top5 / _count
    test_accs_top5.append(test_acc_top5)

    #############################
            
    print ("train acc: %f test acc: %f" % (train_acc, test_acc))
    print ("train acc top 5: %f test acc top 5: %f" % (train_acc_top5, test_acc_top5))
    
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


