
import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
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
ALPHA = args.alpha
sparse = args.sparse
rank = args.rank

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


def parse_function(filename, label):

    # filename = tf.Print(filename, [filename], message="", summarize=100)

    image_string = tf.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=3)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32) * 256.

    image = tf.random_crop(value=image, size=[224, 224, 3])
    image = tf.image.resize_images(image, [224, 224])
    # image = tf.crop_and_resize()

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

    #############################################################

    f = open("/home/bcrafton3/ILSVRC2012/train_labels.txt", 'r')
    lines = f.readlines()
    label_counter = 0
    order_labels = {}
    for line in lines:
        line = line.split()
        order_labels[line[0]] = label_counter
        label_counter += 1
    f.close()

    f = open("/home/bcrafton3/ILSVRC2012/train_labels1.txt", 'r')
    lines = f.readlines()
    label_to_folder = {}
    for line in lines:
        line = line.split()
        label_to_folder[int(line[0])] = line[1]
    f.close()

    f = open('/home/bcrafton3/ILSVRC2012/ILSVRC2012_validation_ground_truth.txt')
    lines = f.readlines()
    for ii in range(len(lines)):
        true_label = int(lines[ii])
        folder = label_to_folder[true_label]
        order_label = order_labels[folder]
        validation_labels.append(order_label)
        # print (true_label, folder, order_label)
    f.close()

    '''
    f = open('/home/bcrafton3/ILSVRC2012/ILSVRC2012_validation_ground_truth.txt')
    lines = f.readlines()
    for ii in range(len(lines)):
        label = int(lines[ii])
        validation_labels.append(label)
    f.close()
    '''

    print (validation_labels[0], validation_images[0])

    #############################################################

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

imgs, labs = get_validation_dataset()
#imgs, labs = get_train_dataset()

filename = tf.placeholder(tf.string, shape=[None])
label_num = tf.placeholder(tf.int64, shape=[None])
dataset = tf.data.Dataset.from_tensor_slices((filename, label_num))
dataset = dataset.shuffle(len(imgs))
dataset = dataset.map(parse_function, num_parallel_calls=4)
dataset = dataset.map(train_preprocess, num_parallel_calls=4)
dataset = dataset.batch(batch_size)
dataset = dataset.repeat()
dataset = dataset.prefetch(8)

iterator = dataset.make_initializable_iterator()
features, labels = iterator.get_next()

features = tf.reshape(features, (-1, 224, 224, 3))
labels = tf.one_hot(labels, depth=num_classes)

###############################################################

#vgg_weights_path='../vgg_weights/vgg_weights.npy'
#vgg_weights_path='vgg_imagenet_weights.npy'
vgg_weights_path='vgg_imagenet_weights_1.npy'

l0 = Convolution(input_sizes=[batch_size, 224, 224, 3], filter_sizes=[3, 3, 3, 64], num_classes=num_classes, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=ALPHA, activation=Relu(), bias=0.0, last_layer=False, name="conv1", load=vgg_weights_path, train=False)
l1 = Convolution(input_sizes=[batch_size, 224, 224, 64], filter_sizes=[3, 3, 64, 64], num_classes=num_classes, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=ALPHA, activation=Relu(), bias=0.0, last_layer=False, name="conv2", load=vgg_weights_path, train=False)
l2 = MaxPool(size=[batch_size, 224, 224, 64], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

l3 = Convolution(input_sizes=[batch_size, 112, 112, 64], filter_sizes=[3, 3, 64, 128], num_classes=num_classes, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=ALPHA, activation=Relu(), bias=0.0, last_layer=False, name="conv3", load=vgg_weights_path, train=False)
l4 = Convolution(input_sizes=[batch_size, 112, 112, 128], filter_sizes=[3, 3, 128, 128], num_classes=num_classes, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=ALPHA, activation=Relu(), bias=0.0, last_layer=False, name="conv4", load=vgg_weights_path, train=False)
l5 = MaxPool(size=[batch_size, 112, 112, 128], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

l6 = Convolution(input_sizes=[batch_size, 56, 56, 128], filter_sizes=[3, 3, 128, 256], num_classes=num_classes, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=ALPHA, activation=Relu(), bias=0.0, last_layer=False, name="conv5", load=vgg_weights_path, train=False)
l7 = Convolution(input_sizes=[batch_size, 56, 56, 256], filter_sizes=[3, 3, 256, 256], num_classes=num_classes, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=ALPHA, activation=Relu(), bias=0.0, last_layer=False, name="conv6", load=vgg_weights_path, train=False)
l8 = Convolution(input_sizes=[batch_size, 56, 56, 256], filter_sizes=[3, 3, 256, 256], num_classes=num_classes, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=ALPHA, activation=Relu(), bias=0.0, last_layer=False, name="conv7", load=vgg_weights_path, train=False)
l9 = MaxPool(size=[batch_size, 56, 56, 256], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

l10 = Convolution(input_sizes=[batch_size, 28, 28, 256], filter_sizes=[3, 3, 256, 512], num_classes=num_classes, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=ALPHA, activation=Relu(), bias=0.0, last_layer=False, name="conv8", load=vgg_weights_path, train=False)
l11 = Convolution(input_sizes=[batch_size, 28, 28, 512], filter_sizes=[3, 3, 512, 512], num_classes=num_classes, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=ALPHA, activation=Relu(), bias=0.0, last_layer=False, name="conv9", load=vgg_weights_path, train=False)
l12 = Convolution(input_sizes=[batch_size, 28, 28, 512], filter_sizes=[3, 3, 512, 512], num_classes=num_classes, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=ALPHA, activation=Relu(), bias=0.0, last_layer=False, name="conv10", load=vgg_weights_path, train=False)
l13 = MaxPool(size=[batch_size, 28, 28, 512], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

l14 = Convolution(input_sizes=[batch_size, 14, 14, 512], filter_sizes=[3, 3, 512, 512], num_classes=num_classes, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=ALPHA, activation=Relu(), bias=0.0, last_layer=False, name="conv11", load=vgg_weights_path, train=False)
l15 = Convolution(input_sizes=[batch_size, 14, 14, 512], filter_sizes=[3, 3, 512, 512], num_classes=num_classes, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=ALPHA, activation=Relu(), bias=0.0, last_layer=False, name="conv12", load=vgg_weights_path, train=False)
l16 = Convolution(input_sizes=[batch_size, 14, 14, 512], filter_sizes=[3, 3, 512, 512], num_classes=num_classes, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=ALPHA, activation=Relu(), bias=0.0, last_layer=False, name="conv13", load=vgg_weights_path, train=False)
l17 = MaxPool(size=[batch_size, 14, 14, 512], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

l18 = ConvToFullyConnected(shape=[7, 7, 512])

#l19 = FullyConnected(size=[7*7*512, 4096], num_classes=num_classes, init_weights=args.init, alpha=ALPHA, activation=Tanh(), bias=0.0, last_layer=False, name="fc1", load=None, train=True)
#l19 = FullyConnected(size=[7*7*512, 4096], num_classes=num_classes, init_weights=args.init, alpha=ALPHA, activation=Relu(), bias=0.0, last_layer=False, name="fc1", load=vgg_weights_path, train=False)
l19 = FullyConnected(size=[7*7*512, 4096], num_classes=num_classes, init_weights=args.init, alpha=ALPHA, activation=Tanh(), bias=0.0, last_layer=False, name="fc1", load=vgg_weights_path, train=False)
#l20 = FeedbackFC(size=[7*7*512, 4096], num_classes=num_classes, sparse=sparse, rank=rank, name="fc1_fb")

#l21 = FullyConnected(size=[4096, 4096], num_classes=num_classes, init_weights=args.init, alpha=ALPHA, activation=Tanh(), bias=0.0, last_layer=False, name="fc2", load=None, train=True)
#l21 = FullyConnected(size=[4096, 4096], num_classes=num_classes, init_weights=args.init, alpha=ALPHA, activation=Relu(), bias=0.0, last_layer=False, name="fc2", load=vgg_weights_path, train=False)
l21 = FullyConnected(size=[4096, 4096], num_classes=num_classes, init_weights=args.init, alpha=ALPHA, activation=Tanh(), bias=0.0, last_layer=False, name="fc2", load=vgg_weights_path, train=False)
#l22 = FeedbackFC(size=[4096, 4096], num_classes=num_classes, sparse=sparse, rank=rank, name="fc2_fb")

#l23 = FullyConnected(size=[4096, num_classes], num_classes=num_classes, init_weights=args.init, alpha=ALPHA, activation=Linear(), bias=0.0, last_layer=True, name="fc3", load=None, train=True)
l23 = FullyConnected(size=[4096, num_classes], num_classes=num_classes, init_weights=args.init, alpha=ALPHA, activation=Linear(), bias=0.0, last_layer=True, name="fc3", load=vgg_weights_path, train=False)

#model = Model(layers=[l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15, l16, l17, l18, l19, l20, l21, l22, l23])
model = Model(layers=[l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15, l16, l17, l18, l19, l21, l23])

#predict = tf.nn.softmax(model.predict(X=features))
predict = model.predict(X=features)
ans = tf.argmax(predict,1)
act = tf.argmax(labels,1)

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

sess.run(iterator.initializer, feed_dict={filename: imgs, label_num: labs})

valid_correct = 0.0
valid_total = 0.0
for j in range(0, len(imgs), batch_size):
    print (j)
    
    [_act, _ans, _total_correct] = sess.run([act, ans, total_correct])
    valid_correct += _total_correct
    valid_total += batch_size

    print (_act)
    print (_ans)
    print ("train accuracy: " + str(valid_correct / valid_total))
    sys.stdout.flush()   
    
    
    
    
    
    

