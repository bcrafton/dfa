
import tensorflow as tf
import numpy as np

rate = 0.25
swap = 0.25
 
shape = (10, 10)
num = int(rate * shape[0] * shape[1])
x = np.zeros(shape=shape)
x = np.reshape(x, (-1))
idxs = np.random.choice(range(0, shape[0] * shape[1]), size=num, replace=False)
x[idxs] = np.random.rand(num)
x = np.reshape(x, shape)

# print (np.count_nonzero(x))

################################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()

weights = tf.placeholder(tf.float32, [10, 10])

################################################

abs_w = tf.abs(weights)
shape = tf.shape(abs_w)
abs_w = tf.reshape(abs_w, (-1,))

vld_i = tf.where(abs_w > 0)
vld_i = tf.reshape(vld_i, (-1,))

vld_w = tf.gather(abs_w, vld_i)

sorted_i = tf.contrib.framework.argsort(vld_w, axis=0)
small_i = tf.gather(vld_i, sorted_i, axis=0)
small_w = tf.gather(abs_w, small_i, axis=0)

################################################

[_abs_w, _vld_i, _vld_w, _sorted_i, _small_i, _small_w] = sess.run([abs_w, vld_i, vld_w, sorted_i, small_i, small_w], feed_dict={weights: x})

# print (np.shape(_abs_w))
# print (np.shape(_vld_i))
# print (np.shape(_vld_w))
# print (np.shape(_sorted_i))
# print (np.shape(_small_i))
# print (np.shape(_small_w))

print (_abs_w)
# print (_vld_i)
# print (_vld_w)
# print (_sorted_w)
print (_small_i)
print (_small_w)

################################################

'''
shape = np.shape(weights)
weights = tf.reshape(weights, (-1))      
  
nswap = self.rate * self.swap

weights_abs = tf.absolute(weights)
valid_idx = tf.where(weights_abs > 0)[0]

smallest_idx = valid_idx[weights_abs[valid_idx].argsort()[0:nswap]]
new_idx = np.random.choice(valid_idx, nswap, replace=False)

sqrt_fan_in = math.sqrt(self.input_size)
new_val = np.random.uniform(low=-1.0/sqrt_fan_in, high=1.0/sqrt_fan_in, size=nswap)

weights[smallest_idx] = 0.
weights[new_idx] = new_val

weights = np.reshape(weights, shape)
idx = np.where(weights > 0)
mask = np.zeros(shape=shape)
mask[idx] = 1

self.weights = tf.Variable(weights, dtype=tf.float32)
self.mask = tf.Variable(mask, dtype=tf.float32)
'''
