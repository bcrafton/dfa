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
parser.add_argument('--batch_size', type=int, default=128)
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
parser.add_argument('--network', type=str, default="local")
args = parser.parse_args()

if args.network == 'local':
    target_size = 224
elif args.network == 'alexnet':
    target_size = 227
else:
    assert(False)

if args.gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

##############################################

import keras
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.local import Conv2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
# from keras.preprocessing import image
# from imagenet_utils import preprocess_input, decode_predictions
from keras import backend as K
import tensorflow as tf

###############################################################

means = [123.68, 116.78, 103.94]
means = np.array(means)
means = np.reshape(means, (1, 1, 3))

# we copied this code from a link below.
# either they had originally used the first index in the image.shape as N or C
# either they expected more than 1 HxWxC image ... or they expected CxHxW
def random_crop(image):
    # print (image.shape)
    height = image.shape[0]
    width = image.shape[1]
    dy = target_size
    dx = target_size
    # if the image is too small we just return original image ? 
    if width < dx or height < dy:
        return image
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    crop = image[y:(y+dy), x:(x+dx), :]
    # print (np.shape(crop))
    return crop

def preprocess(image):
    # image = random_crop(image)
    image = image - means
    return image

###############################################################

# preprocessor function must return same sized image ... cropping here does not work :(
# need to resize and crop or something ... let it resize to 22?x22? ... then we need to crop it and resize it back to 22?.
# do feature wise center manually bc dont want to recalculate each time
# this dude did the random crops after getting the batches from keras: https://jkjung-avt.github.io/keras-image-cropping/
# other useful link: https://github.com/keras-team/keras/issues/3338
# search: ImageDataGenerator random crop

# the datagen is so slow
# train_datagen = ImageDataGenerator(featurewise_center=False, horizontal_flip=True, vertical_flip=True, preprocessing_function=preprocess)

train_datagen = ImageDataGenerator(featurewise_center=False, preprocessing_function=preprocess)

train_generator = train_datagen.flow_from_directory(
    directory='/home/bcrafton3/keras_imagenet/keras_imagenet_train/',
    target_size=(target_size, target_size),
    color_mode="rgb",
    batch_size=args.batch_size,
    class_mode="categorical",
    shuffle=True
    # seed=42
)

###############################################################

val_datagen = ImageDataGenerator(featurewise_center=False, preprocessing_function=preprocess)

val_generator = val_datagen.flow_from_directory(
    directory='/home/bcrafton3/keras_imagenet/keras_imagenet_val/',
    target_size=(target_size, target_size),
    color_mode="rgb",
    batch_size=args.batch_size,
    class_mode="categorical",
    shuffle=True
    # seed=42
)

###############################################################

if args.network == 'local':
    model = Sequential()
    model.add(Conv2D(48, kernel_size=(9, 9), strides=[4, 4], padding="valid", data_format='channels_last', activation='relu', use_bias=True, kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer=keras.initializers.Zeros(), input_shape=(224, 224, 3)))
    model.add(Conv2D(48, kernel_size=(3, 3), strides=[2, 2], padding="valid", data_format='channels_last', activation='relu', use_bias=True, kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer=keras.initializers.Zeros()))

    model.add(Conv2D(96, kernel_size=(5, 5), strides=[1, 1], padding="valid", data_format='channels_last', activation='relu', use_bias=True, kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer=keras.initializers.Zeros()))
    model.add(Conv2D(96, kernel_size=(3, 3), strides=[2, 2], padding="valid", data_format='channels_last', activation='relu', use_bias=True, kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer=keras.initializers.Zeros()))

    model.add(Conv2D(192, kernel_size=(3, 3), strides=[1, 1], padding="valid", data_format='channels_last', activation='relu', use_bias=True, kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer=keras.initializers.Zeros()))
    model.add(Conv2D(192, kernel_size=(3, 3), strides=[2, 2], padding="valid", data_format='channels_last', activation='relu', use_bias=True, kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer=keras.initializers.Zeros()))

    model.add(Conv2D(384, kernel_size=(3, 3), strides=[1, 1], padding="valid", data_format='channels_last', activation='relu', use_bias=True, kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer=keras.initializers.Zeros()))

    model.add(Flatten())

    model.add(Dense(1000, activation='softmax', use_bias=True, kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer=keras.initializers.Zeros()))

elif args.network == 'alexnet':
    model = Sequential()
    model.add(Conv2D(96, kernel_size=(11, 11), strides=[4, 4], padding="valid", data_format='channels_last', activation='relu', use_bias=True, kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer=keras.initializers.Zeros(), input_shape=(227, 227, 3)))
    model.add(MaxPooling2D(pool_size=(3, 3), padding="valid", strides=[2, 2]))

    model.add(Conv2D(256, kernel_size=(5, 5), strides=[1, 1], padding="same", data_format='channels_last', activation='relu', use_bias=True, kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer=keras.initializers.Ones()))
    model.add(MaxPooling2D(pool_size=(3, 3), padding="valid", strides=[2, 2]))

    model.add(Conv2D(384, kernel_size=(3, 3), strides=[1, 1], padding="same", data_format='channels_last', activation='relu', use_bias=True, kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer=keras.initializers.Zeros()))

    model.add(Conv2D(384, kernel_size=(3, 3), strides=[1, 1], padding="same", data_format='channels_last', activation='relu', use_bias=True, kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer=keras.initializers.Ones()))

    model.add(Conv2D(256, kernel_size=(3, 3), strides=[1, 1], padding="same", data_format='channels_last', activation='relu', use_bias=True, kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer=keras.initializers.Ones()))
    model.add(MaxPooling2D(pool_size=(3, 3), padding="valid", strides=[2, 2]))

    model.add(Flatten())

    model.add(Dense(4096, activation='relu', use_bias=True, kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer=keras.initializers.Ones()))
    model.add(Dense(4096, activation='relu', use_bias=True, kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer=keras.initializers.Ones()))
    model.add(Dense(1000, activation='softmax', use_bias=True, kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.01), bias_initializer=keras.initializers.Ones()))

else:
    assert(False)

###############################################################

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1.0), metrics=['accuracy'])

###############################################################

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VAL=val_generator.n//val_generator.batch_size
model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    epochs=args.epochs,
                    verbose=args.verbose,
                    validation_data=val_generator,
                    validation_steps=STEP_SIZE_VAL,
                    workers=8,
                    use_multiprocessing=True
)




