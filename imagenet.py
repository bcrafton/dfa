
import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--alpha', type=float, default=1e-2)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dfa', type=int, default=0)
parser.add_argument('--sparse', type=int, default=0)
parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--init', type=str, default="sqrt_fan_in")
parser.add_argument('--save', type=int, default=0)
args = parser.parse_args()

if args.gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

##############################################

import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

import tensorflow as tf
import os
import math
import numpy
import numpy as np
np.set_printoptions(threshold=1000)

import time
from PIL import Image

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

batch_size = args.batch_size
num_classes = 1000
epochs = args.epochs
data_augmentation = False

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
ALPHA = args.alpha
sparse = args.sparse
rank = args.rank

##############################################

def parse_function(filename, label):
    image_string = tf.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=3)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    image = tf.image.resize_images(image, [227, 227])
    return image, label

def train_preprocess(image, label):
    # image = tf.image.random_flip_left_right(image)

    # image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    # Make sure the image is still in [0, 1]
    # image = tf.clip_by_value(image, 0.0, 1.0)

    return image, label

##############################################

def get_validation_dataset():
    label_counter = 0
    validation_images = []
    validation_labels = []

    print ("building validation dataset")

    for subdir, dirs, files in os.walk('/home/bcrafton3/ILSVRC2012/val/'):
        for file in files:
            validation_images.append(os.path.join('/home/bcrafton3/ILSVRC2012/val/', file))

    validation_labels_file = open('/home/bcrafton3/ILSVRC2012/ILSVRC2012_validation_ground_truth.txt')
    lines = validation_labels_file.readlines()
    for ii in range(len(lines)):
        validation_labels.append(int(lines[ii]))

    print (len(validation_images), len(validation_labels))
    remainder = len(validation_labels) % batch_size
    validation_images = validation_images[:(-remainder)]
    validation_labels = validation_labels[:(-remainder)]

    print("validation data is ready...")

    return validation_images, validation_labels
    
def get_train_dataset():

    label_counter = 0
    training_images = []
    training_labels = []

    print ("making labels dict")

    f = open("/home/bcrafton3/ILSVRC2012/train_labels.txt", 'r')
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

    remainder = len(training_labels) % batch_size
    training_images = training_images[:(-remainder)]
    training_labels = training_labels[:(-remainder)]

    print("Data is ready...")

    return training_images, training_labels

###############################################################

val_imgs, val_labs = get_validation_dataset()

val_dataset = tf.data.Dataset.from_tensor_slices((filename, label_num))
val_dataset = val_dataset.map(parse_function, num_parallel_calls=4)
val_dataset = val_dataset.map(train_preprocess, num_parallel_calls=4)
val_dataset = val_dataset.batch(batch_size)
val_dataset = val_dataset.repeat()
val_dataset = val_dataset.prefetch(8)

###############################################################

train_imgs, train_labs = get_train_dataset()

train_dataset = tf.data.Dataset.from_tensor_slices((filename, label_num))
train_dataset = train_dataset.map(parse_function, num_parallel_calls=4)
train_dataset = train_dataset.map(train_preprocess, num_parallel_calls=4)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.prefetch(8)

###############################################################

handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
features, labels = iterator.get_next()
features = tf.reshape(features, (-1, 227, 227, 3))
labels = tf.one_hot(labels, depth=num_classes)

training_iterator = training_dataset.make_initializable_iterator()
validation_iterator = validation_dataset.make_initializable_iterator()

###############################################################

train = True
weights = None

else:
    l0 = Convolution(input_sizes=[batch_size, 227, 227, 3], filter_sizes=[11, 11, 3, 96], num_classes=num_classes, init_filters=args.init, strides=[1, 4, 4, 1], padding="VALID", alpha=ALPHA, activation=Relu(), bias=0.0, last_layer=False, name="conv1", load=weights, train=train)
    l1 = MaxPool(size=[batch_size, 55, 55, 96], ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")
    # l2 = FeedbackConv(size=[batch_size, 27, 27, 96], num_classes=num_classes, sparse=sparse, rank=rank, name="conv1_fb")

    l3 = Convolution(input_sizes=[batch_size, 27, 27, 96], filter_sizes=[5, 5, 96, 256], num_classes=num_classes, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=ALPHA, activation=Relu(), bias=0.0, last_layer=False, name="conv2", load=weights, train=train)
    l4 = MaxPool(size=[batch_size, 27, 27, 256], ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")
    # l5 = FeedbackConv(size=[batch_size, 13, 13, 256], num_classes=num_classes, sparse=sparse, rank=rank, name="conv2_fb")

    l6 = Convolution(input_sizes=[batch_size, 13, 13, 256], filter_sizes=[3, 3, 256, 384], num_classes=num_classes, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=ALPHA, activation=Relu(), bias=0.0, last_layer=False, name="conv3", load=weights, train=train)
    # l7 = FeedbackConv(size=[batch_size, 13, 13, 384], num_classes=num_classes, sparse=sparse, rank=rank, name="conv3_fb")

    l8 = Convolution(input_sizes=[batch_size, 13, 13, 384], filter_sizes=[3, 3, 384, 384], num_classes=num_classes, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=ALPHA, activation=Relu(), bias=0.0, last_layer=False, name="conv4", load=weights, train=train)
    # l9 = FeedbackConv(size=[batch_size, 13, 13, 384], num_classes=num_classes, sparse=sparse, rank=rank, name="conv4_fb")

    l10 = Convolution(input_sizes=[batch_size, 13, 13, 384], filter_sizes=[3, 3, 384, 256], num_classes=num_classes, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=ALPHA, activation=Relu(), bias=0.0, last_layer=False, name="conv5", load=weights, train=train)
    l11 = MaxPool(size=[batch_size, 13, 13, 256], ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")
    # l12 = FeedbackConv(size=[batch_size, 6, 6, 256], num_classes=num_classes, sparse=sparse, rank=rank, name="conv5_fb")

    l13 = ConvToFullyConnected(shape=[6, 6, 256])
    l14 = FullyConnected(size=[6*6*256, 4096], num_classes=num_classes, init_weights=args.init, alpha=ALPHA, activation=Relu(), bias=0.0, last_layer=False, name="fc1", load=weights, train=train)
    # l15 = FeedbackFC(size=[6*6*256, 4096], num_classes=num_classes, sparse=sparse, rank=rank, name="fc1_fb")

    l16 = FullyConnected(size=[4096, 4096], num_classes=num_classes, init_weights=args.init, alpha=ALPHA, activation=Relu(), bias=0.0, last_layer=False, name="fc2", load=weights, train=train)
    # l17 = FeedbackFC(size=[4096, 4096], num_classes=num_classes, sparse=sparse, rank=rank, name="fc2_fb")

    l18 = FullyConnected(size=[4096, num_classes], num_classes=num_classes, init_weights=args.init, alpha=ALPHA, activation=Relu(), bias=0.0, last_layer=True, name="fc3", load=weights, train=train)

###############################################################

# model = Model(layers=[l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15, l16, l17, l18])
model = Model(layers=[l0, l1, l3, l4, l6, l8, l10, l11, l13, l14, l16, l18])

predict = tf.nn.softmax(model.predict(X=features))

if args.dfa:
    train = model.dfa(X=features, Y=labels)
else:
    train = model.train(X=features, Y=labels)

correct = tf.equal(tf.argmax(predict,1), tf.argmax(labels,1))
total_correct = tf.reduce_sum(tf.cast(correct, tf.float32))
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

weights = model.get_weights()

print (model.num_params())

###############################################################

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.InteractiveSession(config=config)
tf.global_variables_initializer().run()

training_handle = sess.run(training_iterator.string_handle())
validation_handle = sess.run(validation_iterator.string_handle())

for i in range(0, epochs):

    sess.run(training_iterator.initializer)
    train_correct = 0.0
    train_total = 0.0
    for j in range(0, len(train_imgs), batch_size):
        print (j)
        
        _total_correct, _ = sess.run([total_correct, train])
        train_correct += _total_correct
        train_total += batch_size

        print ("train accuracy: " + str(train_correct / train_total))        
    
    sess.run(validation_iterator.initializer)
    val_correct = 0.0
    val_total = 0.0
    for j in range(0, len(val_imgs), batch_size):
        print (j)
        _total_correct = sess.run([total_correct])
        val_correct += _total_correct
        val_total += batch_size

    if args.save:
        [w] = sess.run([weights], feed_dict={})
        np.save("imagenet_weights", w)

    print('epoch {}/{}'.format(i, epochs))
    
    
    
    
    
    
    
    
    

