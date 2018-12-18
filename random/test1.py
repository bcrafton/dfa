
import numpy as np
import tensorflow as tf

x = np.load('cifar10_conv_weights.npy').item()

print (np.std(x['fc1']), np.std(x['fc1_fb']))
print (np.std(x['fc2']), np.std(x['fc2_fb']))
print (np.std(x['fc3']))

print (np.std(x['conv1']), np.std(x['conv1_fb']))
print (np.std(x['conv2']), np.std(x['conv2_fb']))
print (np.std(x['conv3']), np.std(x['conv3_fb']))

print (np.shape(x['conv1']))
print (np.shape(x['conv2']))
print (np.shape(x['conv3']))

########################################################

fc2_fb = x['fc3']
fc1_fb = np.dot(x['fc2'], x['fc3'])
conv3_fb = np.dot(x['fc1'], np.dot(x['fc2'], x['fc3']))

'''
print (np.shape(fc2_fb))
print (np.shape(fc1_fb))
print (np.shape(conv3_fb)) 

print (np.shape(x['conv3']))
print (np.shape(x['conv2']))
'''

########################################################

'''
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()

conv2_fb = tf.nn.conv2d(input=conv3_fb, filter=x['conv3'], strides=[1,1,1,1], padding="SAME")
conv1_fb = tf.nn.conv2d(input=conv2_fb, filter=x['conv2'], strides=[1,1,1,1], padding="SAME")

[_conv2_fb, _conv1_fb] = sess.run([conv2_fb, conv1_fb])
'''
