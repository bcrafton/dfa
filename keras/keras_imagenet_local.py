from __future__ import print_function

import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--alpha', type=float, default=1e-2)
parser.add_argument('--decay', type=float, default=1.)
parser.add_argument('--eps', type=float, default=1.)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--act', type=str, default='tanh')
parser.add_argument('--bias', type=float, default=0.)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dfa', type=int, default=0)
parser.add_argument('--sparse', type=int, default=0)
parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--init', type=str, default="alexnet")
parser.add_argument('--opt', type=str, default="adam")
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--name', type=str, default="imagenet_alexnet")
parser.add_argument('--load', type=str, default=None)
parser.add_argument('--verbose', type=int, default=1)
args = parser.parse_args()

if args.gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

##############################################

import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.local import LocallyConnected2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
# from keras.preprocessing import image
# from imagenet_utils import preprocess_input, decode_predictions
from keras import backend as K
import tensorflow as tf

##############################################
    
def get_train_dataset():

    label_counter = 0
    training_images = []
    training_labels = []

    print ("making labels dict")

    f = open('/home/bcrafton3/dfa/imagenet_labels/train_labels.txt', 'r')
    lines = f.readlines()

    labels = {}
    for line in lines:
        line = line.split(' ')
        labels[line[0]] = label_counter
        label_counter += 1

    f.close()

    print ("building dataset")

    for subdir, dirs, files in os.walk('/home/bcrafton3/ILSVRC2012/train/'):
        for folder in dirs:
            for folder_subdir, folder_dirs, folder_files in os.walk(os.path.join(subdir, folder)):
                for file in folder_files:
                    training_images.append(os.path.join(folder_subdir, file))
                    training_labels.append(labels[folder])

    remainder = len(training_labels) % args.batch_size
    training_images = training_images[:(-remainder)]
    training_labels = training_labels[:(-remainder)]

    print("Data is ready...")

    return training_images, training_labels

###############################################################

train_imgs, train_labs = get_train_dataset()

###############################################################

model = Sequential()
model.add(LocallyConnected2D(48, kernel_size=(9, 9), strides=[4, 4], padding="valid", data_format='channels_last', activation='relu', use_bias=True, kernel_initializer='glorot_normal', input_shape=(224, 224, 3)))
model.add(LocallyConnected2D(48, kernel_size=(3, 3), strides=[2, 2], padding="valid", data_format='channels_last', activation='relu', use_bias=True, kernel_initializer='glorot_normal'))

model.add(LocallyConnected2D(96, kernel_size=(5, 5), strides=[1, 1], padding="valid", data_format='channels_last', activation='relu', use_bias=True, kernel_initializer='glorot_normal'))
model.add(LocallyConnected2D(96, kernel_size=(3, 3), strides=[2, 2], padding="valid", data_format='channels_last', activation='relu', use_bias=True, kernel_initializer='glorot_normal'))

model.add(LocallyConnected2D(192, kernel_size=(3, 3), strides=[1, 1], padding="valid", data_format='channels_last', activation='relu', use_bias=True, kernel_initializer='glorot_normal'))
model.add(LocallyConnected2D(192, kernel_size=(3, 3), strides=[2, 2], padding="valid", data_format='channels_last', activation='relu', use_bias=True, kernel_initializer='glorot_normal'))

model.add(LocallyConnected2D(384, kernel_size=(3, 3), strides=[1, 1], padding="valid", data_format='channels_last', activation='relu', use_bias=True, kernel_initializer='glorot_normal'))

model.add(Flatten())

model.add(Dense(1000, activation='softmax'))

###############################################################

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

###############################################################

for ii in range(0, args.epochs):
    for jj in range(0, len(train_imgs), args.batch_size):
        start = jj
        end = jj + args.batch_size
        x_train = load_img(train_imgs[start], target_size=(224, 224))
        y_train = train_labs[start]
        model.fit(x_train, y_train, batch_size=args.batch_size, epochs=args.epochs, verbose=args.verbose)







