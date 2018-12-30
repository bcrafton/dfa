from __future__ import print_function

import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--alpha', type=float, default=1e-2)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--verbose', type=int, default=1)
args = parser.parse_args()

if args.gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

##############################################

import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.local import LocallyConnected2D
from keras import backend as K
from keras import regularizers
import tensorflow as tf

BATCH_SIZE = 64
NUM_CLASSES = 10
EPOCHS = args.epochs
TRAINING_EXAMPLES = 50000
TESTING_EXAMPLES = 10000
learning_rate = args.alpha

cifar10 = tf.keras.datasets.cifar10.load_data()

(x_train, y_train), (x_test, y_test) = cifar10

#####################################################

# assert(np.shape(x_train) == (TRAINING_EXAMPLES, 32, 32, 3))
# assert(np.shape(x_test) == (TESTING_EXAMPLES, 32, 32, 3))

# print (np.shape(x_train), np.shape(x_test))

if np.shape(x_train) == (TRAINING_EXAMPLES, 3, 32, 32):
    x_train = np.transpose(x_train, (0, 2, 3, 1))
if np.shape(x_train) == (TESTING_EXAMPLES, 3, 32, 32):
    x_test = np.transpose(x_test, (0, 2, 3, 1))

assert(np.shape(x_train) == (TRAINING_EXAMPLES, 32, 32, 3))
assert(np.shape(x_test) == (TESTING_EXAMPLES, 32, 32, 3))

#####################################################

x_train = x_train.reshape(TRAINING_EXAMPLES, 32, 32, 3)
x_train = x_train - np.average(x_train)
x_train = x_train / np.std(x_train)
y_train = keras.utils.to_categorical(y_train, 10)

x_test = x_test.reshape(TESTING_EXAMPLES, 32, 32, 3)
x_test = x_test - np.average(x_test)
x_test = x_test / np.std(x_test)
y_test = keras.utils.to_categorical(y_test, 10)

model = Sequential()

model.add(LocallyConnected2D(96, kernel_size=(5, 5), strides=[2, 2], padding="valid", data_format='channels_last', activation='relu', use_bias=True, kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(0.01), input_shape=(32, 32, 3)))
# model.add(MaxPooling2D(pool_size=(3, 3), padding="same", strides=[2, 2]))
model.add(Dropout(0.0))

model.add(LocallyConnected2D(128, kernel_size=(5, 5), strides=[2, 2], padding="valid", data_format='channels_last', activation='relu', use_bias=True, kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(0.01)))
# model.add(MaxPooling2D(pool_size=(3, 3), padding="same", strides=[2, 2]))
model.add(Dropout(0.0))

model.add(LocallyConnected2D(256, kernel_size=(5, 5), strides=[2, 2], padding="valid", data_format='channels_last', activation='relu', use_bias=True, kernel_initializer='glorot_normal', kernel_regularizer=regularizers.l2(0.01)))
# model.add(MaxPooling2D(pool_size=(3, 3), padding="same", strides=[2, 2]))
model.add(Dropout(0.0))

model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(x_train, 
          y_train,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          verbose=args.verbose,
          validation_data=(x_test, y_test))
          
score = model.evaluate(x_test, y_test, verbose=0)

print('Test loss:', score[0])
print('Test accuracy:', score[1])
