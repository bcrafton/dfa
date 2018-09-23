
import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--alpha', type=float, default=1e-2)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dfa', type=int, default=1)
parser.add_argument('--sparse', type=int, default=1)
parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--init', type=str, default="NA")
parser.add_argument('--opt', type=str, default="NA")
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

label_counter = 0

training_images = []
training_labels = []

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

print ("building dataset")

for subdir, dirs, files in os.walk('/home/bcrafton3/ILSVRC2012/train/'):
    for folder in dirs:
        for folder_subdir, folder_dirs, folder_files in os.walk(os.path.join(subdir, folder)):
            for file in folder_files:
                training_images.append(os.path.join(folder_subdir, file))
                # training_labels.append( keras.utils.to_categorical(label_counter, num_classes) )
                
                # label = np.zeros(num_classes)
                # label[label_counter] = 1
                # training_labels.append(label)

                training_labels.append(label_counter)

        label_counter = label_counter + 1
        print (str(label_counter) + "/" + str(num_classes))

remainder = len(training_labels) % batch_size
training_images = training_images[:(-remainder)]
training_labels = training_labels[:(-remainder)]

filename = tf.placeholder(tf.string, shape=[None])
label_num = tf.placeholder(tf.int64, shape=[None])
dataset = tf.data.Dataset.from_tensor_slices((filename, label_num))
dataset = dataset.shuffle(len(training_images))
dataset = dataset.map(parse_function, num_parallel_calls=4)
dataset = dataset.map(train_preprocess, num_parallel_calls=4)
dataset = dataset.batch(batch_size)
dataset = dataset.repeat()
dataset = dataset.prefetch(8)

print("Data is ready...")

iterator = dataset.make_initializable_iterator()
features, labels = iterator.get_next()

features = tf.reshape(features, (-1, 227, 227, 3))
labels = tf.one_hot(labels, depth=num_classes)

###############################################################

args_ext = "_dfa_" + str(args.dfa) + "_sparse_" + str(args.sparse)

l0 = Convolution(input_sizes=[batch_size, 227, 227, 3], filter_sizes=[11, 11, 3, 96], num_classes=num_classes, init_filters=args.init, strides=[1, 4, 4, 1], padding="VALID", alpha=ALPHA, activation=Tanh(), bias=0.0, last_layer=False, name="conv1" + args_ext)
l1 = MaxPool(size=[batch_size, 55, 55, 96], ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")
l2 = FeedbackConv(size=[batch_size, 27, 27, 96], num_classes=num_classes, sparse=sparse, rank=rank, name="conv_fb1" + args_ext)

l3 = Convolution(input_sizes=[batch_size, 27, 27, 96], filter_sizes=[5, 5, 96, 256], num_classes=num_classes, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=ALPHA, activation=Tanh(), bias=0.0, last_layer=False, name="conv2" + args_ext)
l4 = MaxPool(size=[batch_size, 27, 27, 256], ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")
l5 = FeedbackConv(size=[batch_size, 13, 13, 256], num_classes=num_classes, sparse=sparse, rank=rank, name="conv_fb2" + args_ext)

l6 = Convolution(input_sizes=[batch_size, 13, 13, 256], filter_sizes=[3, 3, 256, 384], num_classes=num_classes, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=ALPHA, activation=Tanh(), bias=0.0, last_layer=False, name="conv3" + args_ext)
l7 = FeedbackConv(size=[batch_size, 13, 13, 384], num_classes=num_classes, sparse=sparse, rank=rank, name="conv_fb3" + args_ext)

l8 = Convolution(input_sizes=[batch_size, 13, 13, 384], filter_sizes=[3, 3, 384, 384], num_classes=num_classes, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=ALPHA, activation=Tanh(), bias=0.0, last_layer=False, name="conv4" + args_ext)
l9 = FeedbackConv(size=[batch_size, 13, 13, 384], num_classes=num_classes, sparse=sparse, rank=rank, name="conv_fb4" + args_ext)

l10 = Convolution(input_sizes=[batch_size, 13, 13, 384], filter_sizes=[3, 3, 384, 256], num_classes=num_classes, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=ALPHA, activation=Tanh(), bias=0.0, last_layer=False, name="conv5" + args_ext)
l11 = MaxPool(size=[batch_size, 13, 13, 256], ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding="VALID")
l12 = FeedbackConv(size=[batch_size, 6, 6, 256], num_classes=num_classes, sparse=sparse, rank=rank, name="conv_fb5" + args_ext)

l13 = ConvToFullyConnected(shape=[6, 6, 256])
l14 = FullyConnected(size=[6*6*256, 4096], num_classes=num_classes, init_weights=args.init, alpha=ALPHA, activation=Tanh(), bias=0.0, last_layer=False, name="fc1" + args_ext)
l15 = FeedbackFC(size=[6*6*256, 4096], num_classes=num_classes, sparse=sparse, rank=rank, name="fc_fb1" + args_ext)

l16 = FullyConnected(size=[4096, 4096], num_classes=num_classes, init_weights=args.init, alpha=ALPHA, activation=Tanh(), bias=0.0, last_layer=False, name="fc2" + args_ext)
l17 = FeedbackFC(size=[4096, 4096], num_classes=num_classes, sparse=sparse, rank=rank, name="fc_fb2" + args_ext)

l18 = FullyConnected(size=[4096, num_classes], num_classes=num_classes, init_weights=args.init, alpha=ALPHA, activation=Linear(), bias=0.0, last_layer=True, name="fc3" + args_ext)

model = Model(layers=[l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15, l16, l17, l18])

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

sess.run(iterator.initializer, feed_dict={filename: training_images, label_num: training_labels})

for i in range(0, epochs):
    train_correct = 0.0
    train_total = 0.0
    for j in range(0, len(training_images), batch_size):
        print (j)
        
        pred, _total_correct, _ = sess.run([predict, total_correct, train])
        train_correct += _total_correct
        train_total += batch_size

        print ("train accuracy: " + str(train_correct / train_total))
        sys.stdout.flush()
        

    if args.save:
        [w] = sess.run([weights], feed_dict={})
        n = model.get_names()
        print (len(w), len(n))
        assert(len(w) == len(n))
        for ii in range(len(w)):
            np.save(n[ii], w[ii])
        
    print('epoch {}/{}'.format(i, epochs))
    
    
    
    
    
    
    
    
    

