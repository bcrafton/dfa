
import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--alpha', type=float, default=1e-2)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dfa', type=int, default=0)
parser.add_argument('--sparse', type=int, default=0)
parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--init', type=str, default="sqrt_fan_in")
parser.add_argument('--opt', type=str, default="adam")
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--name', type=str, default="cifar10_fc_weights")
parser.add_argument('--num', type=int, default=0)
parser.add_argument('--angles', type=int, default=0)
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
ALPHA = args.alpha
sparse = args.sparse
rank = args.rank

##############################################

tf.set_random_seed(0)
tf.reset_default_graph()

batch_size = tf.placeholder(tf.int32, shape=())
dropout_rate = tf.placeholder(tf.float32, shape=())
learning_rate = tf.placeholder(tf.float32, shape=())
#XTRAIN = tf.placeholder(tf.float32, [None, 32, 32, 3])
YTRAIN = tf.placeholder(tf.float32, [None, 10])
#XTRAIN = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), XTRAIN)
#XTRAIN = tf.reshape(XTRAIN, [batch_size, 3072])
XTRAIN = tf.placeholder(tf.float32, [None, 3072])

#XTEST = tf.placeholder(tf.float32, [None, 32, 32, 3])
YTEST = tf.placeholder(tf.float32, [None, 10])
#XTEST = tf.map_fn(lambda frame1: tf.image.per_image_standardization(frame1), XTEST)
#XTEST = tf.reshape(XTEST, [batch_size, 3072])
XTEST = tf.placeholder(tf.float32, [None, 3072])

l0 = FullyConnected(size=[3072, 1000], num_classes=10, init_weights=args.init, alpha=ALPHA, activation=Tanh(), bias=0.0, last_layer=False, name="fc1")
l1 = Dropout(rate=dropout_rate/4.)
l2 = FeedbackFC(size=[3072, 1000], num_classes=10, sparse=sparse, rank=rank, name="fc1_fb")

l3 = FullyConnected(size=[1000, 1000], num_classes=10, init_weights=args.init, alpha=ALPHA, activation=Tanh(), bias=0.0, last_layer=False, name="fc2")
l4 = Dropout(rate=dropout_rate/2.)
l5 = FeedbackFC(size=[1000, 1000], num_classes=10, sparse=sparse, rank=rank, name="fc2_fb")

l6 = FullyConnected(size=[1000, 1000], num_classes=10, init_weights=args.init, alpha=ALPHA, activation=Tanh(), bias=0.0, last_layer=False, name="fc3")
l7 = Dropout(rate=dropout_rate)
l8 = FeedbackFC(size=[1000, 1000], num_classes=10, sparse=sparse, rank=rank, name="fc3_fb")

l9 = FullyConnected(size=[1000, 10], num_classes=10, init_weights=args.init, alpha=ALPHA, activation=Linear(), bias=0.0, last_layer=True, name="fc4")

model = Model(layers=[l0, l1, l2, l3, l4, l5, l6, l7, l8, l9])

##############################################

predict = model.predict(X=XTEST)

weights = model.get_weights()

if args.opt == "adam":
    if args.dfa:
        grads_and_vars = model.dfa_gvs(X=XTRAIN, Y=YTRAIN)
    else:
        grads_and_vars = model.gvs(X=XTRAIN, Y=YTRAIN)
        
    train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).apply_gradients(grads_and_vars=grads_and_vars)
    # train = tf.train.RMSPropOptimizer(learning_rate=ALPHA, decay=0.975, epsilon=1.0).apply_gradients(grads_and_vars=grads_and_vars)

else:
    if args.dfa:
        train = model.dfa(X=XTRAIN, Y=YTRAIN)
    else:
        train = model.train(X=XTRAIN, Y=YTRAIN)

correct_prediction = tf.equal(tf.argmax(predict,1), tf.argmax(YTEST,1))
correct_prediction_sum = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

##############################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()

(x_train, y_train), (x_test, y_test) = cifar10

mean = np.mean(x_train, axis=(0, 1, 2), keepdims=True)
std = np.std(x_train, axis=(0, 1, 2), keepdims=True)

# x_train = x_train / 255.
x_train = (x_train - mean) / std
x_train = x_train.reshape(TRAIN_EXAMPLES, 3072)
y_train = keras.utils.to_categorical(y_train, 10)

# x_test = x_test / 255.
x_test = (x_test - mean) / std
x_test = x_test.reshape(TEST_EXAMPLES, 3072)
y_test = keras.utils.to_categorical(y_test, 10)
##############################################

filename = "cifar10_fc_" +              \
           str(args.epochs) + "_" +     \
           str(args.batch_size) + "_" + \
           str(args.alpha) + "_" +      \
           str(args.dfa) + "_" +        \
           str(args.sparse) + "_" +     \
           str(args.gpu) + "_" +        \
           args.init + "_" +            \
           args.opt +                   \
           ".results"

f = open(filename, "w")
f.write(filename + "\n")
f.write("total params: " + str(model.num_params()) + "\n")
f.close()

##############################################

for ii in range(EPOCHS):
    decay = np.power(0.995, ii)
    lr = ALPHA * decay
    print (ii, lr)
    for jj in range(0, int(TRAIN_EXAMPLES/BATCH_SIZE) * BATCH_SIZE, BATCH_SIZE):
        start = jj % TRAIN_EXAMPLES
        end = jj % TRAIN_EXAMPLES + BATCH_SIZE
        sess.run([train], feed_dict={batch_size: BATCH_SIZE, dropout_rate: 0.5, learning_rate: lr, XTRAIN: x_train[start:end], YTRAIN: y_train[start:end]})
    
    count = 0
    total_correct = 0
    
    for jj in range(0, int(TEST_EXAMPLES/BATCH_SIZE) * BATCH_SIZE, BATCH_SIZE):
        start = jj % TEST_EXAMPLES
        end = jj % TEST_EXAMPLES + BATCH_SIZE
        correct = sess.run(correct_prediction_sum, feed_dict={batch_size: BATCH_SIZE, dropout_rate: 0.0, learning_rate: 0.0, XTEST: x_test[start:end], YTEST: y_test[start:end]})

        count += BATCH_SIZE
        total_correct += correct

    print (total_correct * 1.0 / count)
    sys.stdout.flush()
    
    f = open(filename, "a")
    f.write(str(total_correct * 1.0 / count) + "\n")
    f.close()

    if args.save:
        [w] = sess.run([weights], feed_dict={})
        np.save(args.name, w)

##############################################

