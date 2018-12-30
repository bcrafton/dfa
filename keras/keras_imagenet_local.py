from __future__ import print_function

import argparse
import os
import sys

##############################################

# https://github.com/fchollet/deep-learning-models
# https://github.com/keras-team/keras/issues/6486
# https://github.com/wichtounet/frameworks/blob/master/keras/experiment6.py
# https://medium.com/@vijayabhaskar96/tutorial-image-classification-with-keras-flow-from-directory-and-generators-95f75ebe5720
# https://keras.io/preprocessing/image/
# flow_from_directory

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--alpha', type=float, default=1e-2)
parser.add_argument('--decay', type=float, default=1.)
parser.add_argument('--eps', type=float, default=1.)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--act', type=str, default='relu')
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

###############################################################

train_datagen = ImageDataGenerator()

train_generator = train_datagen.flow_from_directory(
    directory='/home/bcrafton3/keras_imagenet/keras_imagenet_train/',
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=args.batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

###############################################################

val_datagen = ImageDataGenerator()

val_generator = val_datagen.flow_from_directory(
    directory='/home/bcrafton3/keras_imagenet/keras_imagenet_val/',
    target_size=(224, 224),
    color_mode="rgb",
    batch_size=args.batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42
)

###############################################################

model = Sequential()
model.add(LocallyConnected2D(48, kernel_size=(9, 9), strides=[4, 4], padding="valid", data_format='channels_last', activation='relu', use_bias=True, kernel_initializer='glorot_normal', input_shape=(224, 224, 3)))
model.add(LocallyConnected2D(48, kernel_size=(3, 3), strides=[2, 2], padding="valid", data_format='channels_last', activation='relu', use_bias=True, kernel_initializer='glorot_normal'))

# model.add(LocallyConnected2D(96, kernel_size=(5, 5), strides=[1, 1], padding="valid", data_format='channels_last', activation='relu', use_bias=True, kernel_initializer='glorot_normal'))
model.add(LocallyConnected2D(96, kernel_size=(3, 3), strides=[2, 2], padding="valid", data_format='channels_last', activation='relu', use_bias=True, kernel_initializer='glorot_normal'))

# model.add(LocallyConnected2D(192, kernel_size=(3, 3), strides=[1, 1], padding="valid", data_format='channels_last', activation='relu', use_bias=True, kernel_initializer='glorot_normal'))
model.add(LocallyConnected2D(192, kernel_size=(3, 3), strides=[2, 2], padding="valid", data_format='channels_last', activation='relu', use_bias=True, kernel_initializer='glorot_normal'))

model.add(LocallyConnected2D(384, kernel_size=(3, 3), strides=[1, 1], padding="valid", data_format='channels_last', activation='relu', use_bias=True, kernel_initializer='glorot_normal'))

model.add(Flatten())

model.add(Dense(1000, activation='softmax'))

###############################################################

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

###############################################################

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VAL=val_generator.n//val_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=val_generator,
                    validation_steps=STEP_SIZE_VAL,
                    epochs=10
)




