import tensorflow as tf
import numpy as np
import keras
import matplotlib.pyplot as plt
from PIL import Image
np.set_printoptions(threshold=np.inf)

#########################################################

kernel = tf.Variable(tf.random_uniform(shape=[100, 10], minval=-1., maxval=1.))

#########################################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()

#####################################################

l2_loss = tf.nn.l2_loss(kernel)

#####################################################

[_l2_loss] = sess.run([l2_loss], feed_dict={})

print (_l2_loss)

#####################################################





