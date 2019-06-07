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

import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras import backend as K
import tensorflow as tf

BATCH_SIZE = 64
NUM_CLASSES = 10
EPOCHS = args.epochs
TRAINING_EXAMPLES = 50000
TESTING_EXAMPLES = 10000
learning_rate = args.alpha

cifar10 = tf.keras.datasets.cifar10.load_data()

(x_train, y_train), (x_test, y_test) = cifar10
assert(np.shape(x_train) == (TRAINING_EXAMPLES, 32, 32, 3))
assert(np.shape(x_test) == (TESTING_EXAMPLES, 32, 32, 3))

x_train = x_train.reshape(TRAINING_EXAMPLES, 32, 32, 3)
mean = np.mean(x_train, axis=0, keepdims=True)
std = np.std(x_train, axis=0, ddof=1, keepdims=True)
x_train = (x_train - mean) / std
y_train = keras.utils.to_categorical(y_train, 10)

x_test = x_test.reshape(TESTING_EXAMPLES, 32, 32, 3)
mean = np.mean(x_test, axis=0, keepdims=True)
std = np.std(x_test, axis=0, ddof=1, keepdims=True)
x_test = (x_test - mean) / std
y_test = keras.utils.to_categorical(y_test, 10)

model = Sequential()
model.add(Conv2D(96, kernel_size=(5, 5), padding="same", activation='relu', input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3), padding="same", strides=[2, 2]))

model.add(Conv2D(128, kernel_size=(5, 5), padding="same", activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3), padding="same", strides=[2, 2]))

model.add(Conv2D(256, kernel_size=(5, 5), padding="same", activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(3, 3), padding="same", strides=[2, 2]))

model.add(Flatten())

model.add(Dense(2048, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(2048, activation='relu'))
model.add(BatchNormalization())
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
