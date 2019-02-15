
import os
import argparse
import tensorflow as tf
import numpy as np

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--lr', type=float, default=1e-2)
args = parser.parse_args()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

LAYER1 = 784
LAYER2 = 400
LAYER3 = 10

TRAIN_EXAMPLES = 60000
TEST_EXAMPLES = 10000

x_train = x_train.reshape(TRAIN_EXAMPLES, 784) 
# assert(np.shape(x_train) == (TRAIN_EXAMPLES, 784))
x_train = x_train.astype('float32')
x_train /= 255.
_y_train = np.zeros(shape=(TRAIN_EXAMPLES, 10))
for ii in range(TRAIN_EXAMPLES):
    _y_train[ii][y_train[ii]] = 1
y_train = _y_train

x_test = x_test.reshape(TEST_EXAMPLES, 784)
# assert(np.shape(x_train) == (TEST_EXAMPLES, 784))
x_test = x_test.astype('float32')
x_test /= 255.
_y_test = np.zeros(shape=(TEST_EXAMPLES, 10))
for ii in range(TEST_EXAMPLES):
    _y_test[ii][y_test[ii]] = 1
y_test = _y_test

EPOCHS = 10
BATCH_SIZE = 32

##############################################

sqrt_fan_in = np.sqrt(784)
weights = tf.Variable(tf.random_uniform(minval=-1./sqrt_fan_in, maxval=1./sqrt_fan_in, shape=[784, 10]))
bias = tf.Variable(tf.zeros(shape=[10]))

##############################################
# forward
##############################################

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

Z = tf.matmul(X, weights) + bias
A = tf.nn.softmax(Z)

##############################################
# backward
##############################################

E = tf.subtract(A, Y)

DW = tf.matmul(tf.transpose(X), E)
DB = tf.reduce_sum(E, axis=0)

weights = weights.assign(tf.subtract(weights, tf.scalar_mul(args.lr, DW)))
bias = bias.assign(tf.subtract(bias, tf.scalar_mul(args.lr, DB)))

##############################################

correct = tf.equal(tf.argmax(A,1), tf.argmax(Y,1))
total_correct = tf.reduce_sum(tf.cast(correct, tf.float32))

##############################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

for epoch in range(args.epochs):
    print ("epoch: %d/%d" % (epoch, args.epochs))

    #############################

    _count = 0
    _total_correct = 0

    for ex in range(0, TRAIN_EXAMPLES, 50):
        start = ex 
        stop = ex + 50
        
        xs = x_train[start:stop]
        ys = y_train[start:stop]
        _correct, _, _ = sess.run([total_correct, weights, bias], feed_dict={X: xs, Y: ys})

        _total_correct += _correct
        _count += 50

    train_acc = 1.0 * _total_correct / _count

    #############################
    
    _correct = sess.run(total_correct, feed_dict={X: x_test, Y: y_test})

    _total_correct = _correct
    _count = TEST_EXAMPLES

    test_acc = 1.0 * _total_correct / _count

    #############################

    print ("train acc: %f test acc: %f" % (train_acc, test_acc))











