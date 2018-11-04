
import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--alpha', type=float, default=1e-2)
parser.add_argument('--decay', type=float, default=0.99)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dfa', type=int, default=0)
parser.add_argument('--sparse', type=int, default=0)
parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--init', type=str, default="sqrt_fan_in")
parser.add_argument('--opt', type=str, default="gd")
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--name', type=str, default="imagenet_vgg")
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

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import scipy.misc

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

##############################################

def _mean_image_subtraction(image, means):
  """Subtracts the given means from each image channel.

  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)

  Note that the rank of `image` must be known.

  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.

  Returns:
    the centered image.

  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
  if image.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')
  num_channels = image.get_shape().as_list()[-1]
  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
  for i in range(num_channels):
    channels[i] = tf.subtract(channels[i], means[i])
  return tf.concat(axis=2, values=channels)


def parse_function_train(filename, label):

    # filename = tf.Print(filename, [filename], message="", summarize=100)

    image_string = tf.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=3)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32) * 256.

    if image.get_shape()[0] >= 224 and image.get_shape()[1] >= 224:
        image = tf.random_crop(value=image, size=[224, 224, 3])
    else:
        image = tf.image.resize_images(image, [224, 224])

    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94
    image = _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])

    return image, label

def parse_function_val(filename, label):

    image_string = tf.read_file(filename)

    image = tf.image.decode_jpeg(image_string, channels=3)

    image = tf.image.resize_images(image, [224, 224])

    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94
    image = _mean_image_subtraction(image, [_R_MEAN, _G_MEAN, _B_MEAN])    

    return image, label

def train_preprocess(image, label):
    # image = tf.image.random_flip_left_right(image)

    # image = tf.image.random_brightness(image, max_delta=32.0 / 255.0)
    # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    # Make sure the image is still in [0, 1]
    # image = tf.clip_by_value(image, 0.0, 1.0)

    # plt.imsave("img.png", image, cmap="gray")
    # assert(False)

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
    validation_images = sorted(validation_images)

    validation_labels_file = open('/home/bcrafton3/dfa/imagenet_labels/validation_labels.txt')
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

    remainder = len(training_labels) % batch_size
    training_images = training_images[:(-remainder)]
    training_labels = training_labels[:(-remainder)]

    print("Data is ready...")

    return training_images, training_labels

###############################################################

filename = tf.placeholder(tf.string, shape=[None])
label = tf.placeholder(tf.int64, shape=[None])

###############################################################

val_imgs, val_labs = get_validation_dataset()

val_dataset = tf.data.Dataset.from_tensor_slices((filename, label))
val_dataset = val_dataset.shuffle(len(val_imgs))
val_dataset = val_dataset.map(parse_function_val, num_parallel_calls=4)
val_dataset = val_dataset.map(train_preprocess, num_parallel_calls=4)
val_dataset = val_dataset.batch(batch_size)
val_dataset = val_dataset.repeat()
val_dataset = val_dataset.prefetch(8)

###############################################################

train_imgs, train_labs = get_train_dataset()

train_dataset = tf.data.Dataset.from_tensor_slices((filename, label))
train_dataset = train_dataset.shuffle(len(train_imgs))
train_dataset = train_dataset.map(parse_function_train, num_parallel_calls=4)
train_dataset = train_dataset.map(train_preprocess, num_parallel_calls=4)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.prefetch(8)

###############################################################

handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
features, labels = iterator.get_next()
features = tf.reshape(features, (-1, 224, 224, 3))
labels = tf.one_hot(labels, depth=num_classes)

train_iterator = train_dataset.make_initializable_iterator()
val_iterator = val_dataset.make_initializable_iterator()

###############################################################

train_conv=False
train_fc=True
weights_conv='../vgg_weights/vgg_weights.npy'
weights_fc=None # '../vgg_weights/vgg_weights.npy'

if args.dfa:
    bias = 0.0
else:
    bias = 0.0

###############################################################

dropout_rate = tf.placeholder(tf.float32, shape=())
learning_rate = tf.placeholder(tf.float32, shape=())

l0 = Convolution(input_sizes=[batch_size, 224, 224, 3], filter_sizes=[3, 3, 3, 64], num_classes=num_classes, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=0.0, last_layer=False, name="conv1", load=weights_conv, train=train_conv)
l1 = Convolution(input_sizes=[batch_size, 224, 224, 64], filter_sizes=[3, 3, 64, 64], num_classes=num_classes, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=0.0, last_layer=False, name="conv2", load=weights_conv, train=train_conv)
l2 = MaxPool(size=[batch_size, 224, 224, 64], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

l3 = Convolution(input_sizes=[batch_size, 112, 112, 64], filter_sizes=[3, 3, 64, 128], num_classes=num_classes, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=0.0, last_layer=False, name="conv3", load=weights_conv, train=train_conv)
l4 = Convolution(input_sizes=[batch_size, 112, 112, 128], filter_sizes=[3, 3, 128, 128], num_classes=num_classes, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=0.0, last_layer=False, name="conv4", load=weights_conv, train=train_conv)
l5 = MaxPool(size=[batch_size, 112, 112, 128], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

l6 = Convolution(input_sizes=[batch_size, 56, 56, 128], filter_sizes=[3, 3, 128, 256], num_classes=num_classes, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=0.0, last_layer=False, name="conv5", load=weights_conv, train=train_conv)
l7 = Convolution(input_sizes=[batch_size, 56, 56, 256], filter_sizes=[3, 3, 256, 256], num_classes=num_classes, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=0.0, last_layer=False, name="conv6", load=weights_conv, train=train_conv)
l8 = Convolution(input_sizes=[batch_size, 56, 56, 256], filter_sizes=[3, 3, 256, 256], num_classes=num_classes, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=0.0, last_layer=False, name="conv7", load=weights_conv, train=train_conv)
l9 = MaxPool(size=[batch_size, 56, 56, 256], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

l10 = Convolution(input_sizes=[batch_size, 28, 28, 256], filter_sizes=[3, 3, 256, 512], num_classes=num_classes, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=0.0, last_layer=False, name="conv8", load=weights_conv, train=train_conv)
l11 = Convolution(input_sizes=[batch_size, 28, 28, 512], filter_sizes=[3, 3, 512, 512], num_classes=num_classes, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=0.0, last_layer=False, name="conv9", load=weights_conv, train=train_conv)
l12 = Convolution(input_sizes=[batch_size, 28, 28, 512], filter_sizes=[3, 3, 512, 512], num_classes=num_classes, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=0.0, last_layer=False, name="conv10", load=weights_conv, train=train_conv)
l13 = MaxPool(size=[batch_size, 28, 28, 512], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

l14 = Convolution(input_sizes=[batch_size, 14, 14, 512], filter_sizes=[3, 3, 512, 512], num_classes=num_classes, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=0.0, last_layer=False, name="conv11", load=weights_conv, train=train_conv)
l15 = Convolution(input_sizes=[batch_size, 14, 14, 512], filter_sizes=[3, 3, 512, 512], num_classes=num_classes, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=0.0, last_layer=False, name="conv12", load=weights_conv, train=train_conv)
l16 = Convolution(input_sizes=[batch_size, 14, 14, 512], filter_sizes=[3, 3, 512, 512], num_classes=num_classes, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=0.0, last_layer=False, name="conv13", load=weights_conv, train=train_conv)
l17 = MaxPool(size=[batch_size, 14, 14, 512], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

l18 = ConvToFullyConnected(shape=[7, 7, 512])
l19 = FullyConnected(size=[7*7*512, 4096], num_classes=num_classes, init_weights=args.init, alpha=learning_rate, activation=Tanh(), bias=bias, last_layer=False, name="fc1", load=weights_fc, train=train_fc)
l20 = Dropout(rate=dropout_rate)
l21 = FeedbackFC(size=[7*7*512, 4096], num_classes=num_classes, sparse=args.sparse, rank=args.rank, name="fc1_fb")

l22 = FullyConnected(size=[4096, 4096], num_classes=num_classes, init_weights=args.init, alpha=learning_rate, activation=Tanh(), bias=bias, last_layer=False, name="fc2", load=weights_fc, train=train_fc)
l23 = Dropout(rate=dropout_rate)
l24 = FeedbackFC(size=[4096, 4096], num_classes=num_classes, sparse=args.sparse, rank=args.rank, name="fc2_fb")

l25 = FullyConnected(size=[4096, num_classes], num_classes=num_classes, init_weights=args.init, alpha=learning_rate, activation=Linear(), bias=bias, last_layer=True, name="fc3", load=weights_fc, train=train_fc)

###############################################################

model = Model(layers=[l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15, l16, l17, l18, l19, l20, l21, l22, l23, l24, l25])

predict = tf.nn.softmax(model.predict(X=features))

if args.opt == "adam" or args.opt == "rms" or args.opt == "decay":
    if args.dfa:
        grads_and_vars = model.dfa_gvs(X=features, Y=labels)
    else:
        grads_and_vars = model.gvs(X=features, Y=labels)
        
    if args.opt == "adam":
        train = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1.0).apply_gradients(grads_and_vars=grads_and_vars)
    elif args.opt == "rms":
        train = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.99, epsilon=1.0).apply_gradients(grads_and_vars=grads_and_vars)
    elif args.opt == "decay":
        train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).apply_gradients(grads_and_vars=grads_and_vars)
    else:
        assert(False)

else:
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

train_handle = sess.run(train_iterator.string_handle())
val_handle = sess.run(val_iterator.string_handle())

###############################################################

filename = args.name + '.results'
f = open(filename, "w")
f.write(filename + "\n")
f.write("total params: " + str(model.num_params()) + "\n")
f.close()

###############################################################

train_accs = []
val_accs = []

for ii in range(0, epochs):

    if args.opt == 'decay' or args.opt == 'gd':
        decay = np.power(args.decay, ii)
        lr = args.alpha * decay
    else:
        lr = args.alpha
        
    print (ii)

    sess.run(train_iterator.initializer, feed_dict={filename: train_imgs, label: train_labs})
    train_correct = 0.0
    train_total = 0.0
    # for j in range(0, 32 * 10, batch_size):
    for j in range(0, len(train_imgs), batch_size):
        print (j)
        
        _total_correct, _ = sess.run([total_correct, train], feed_dict={handle: train_handle, dropout_rate: 0.5, learning_rate: lr})
        train_correct += _total_correct
        train_total += batch_size
        train_acc = train_correct / train_total
        
        print ("train accuracy: " + str(train_acc))        
        f = open(filename, "a")
        f.write(str(train_acc) + "\n")
        f.close()

    train_accs.append(train_acc)
    
    sess.run(val_iterator.initializer, feed_dict={filename: val_imgs, label: val_labs})
    val_correct = 0.0
    val_total = 0.0
    lr = 0.0
    # for j in range(0, 32 * 10, batch_size):
    for j in range(0, len(val_imgs), batch_size):
        print (j)

        [_total_correct] = sess.run([total_correct], feed_dict={handle: val_handle, dropout_rate: 0.0, learning_rate: lr})
        val_correct += _total_correct
        val_total += batch_size
        val_acc = val_correct / val_total
        
        print ("val accuracy: " + str(val_acc))
        f = open(filename, "a")
        f.write(str(val_acc) + "\n")
        f.close()

    val_accs.append(val_acc)

    if args.save:
        [w] = sess.run([weights], feed_dict={handle: val_handle, dropout_rate: 0.0, learning_rate: lr})
        w['train_acc'] = train_accs
        w['val_acc'] = val_accs
        np.save(args.name, w)

    print('epoch {}/{}'.format(ii, epochs))
    

