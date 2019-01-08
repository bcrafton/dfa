
import tensorflow as tf
import numpy as np

rate = 0.25
swap = 0.25
 
shape = (5, 5, 3, 96)
mask = np.random.choice([0, 1], size=shape, replace=True, p=[1.-rate, rate])
num = np.count_nonzero(mask)
nswap = int(num * swap)
slice_size = num - nswap
x = np.random.uniform(low=-1., high=1., size=shape) * mask

################################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()

weights = tf.placeholder(tf.float32, [5, 5, 3, 96])

################################################

shape = tf.shape(weights)
abs_w = tf.abs(weights)
vld_i = tf.where(abs_w > 0)
vld_w = tf.gather_nd(abs_w, vld_i)
sorted_i = tf.contrib.framework.argsort(vld_w, axis=0)

# new indices
new_i = tf.where(abs_w <= 0)
new_i = tf.random_shuffle(new_i)
new_i = tf.slice(new_i, [0, 0], [nswap, 4])
new_i = tf.cast(new_i, tf.int32)
sqrt_fan_in = np.sqrt(32 * 32 * 3)
new_w = tf.random_uniform(minval=-1.0/sqrt_fan_in, maxval=1.0/sqrt_fan_in, shape=(nswap,))

# largest indices (rate - rate * nswap)
large_i = tf.gather(vld_i, sorted_i, axis=0)
large_i = tf.cast(large_i, tf.int32)
large_i = tf.slice(large_i, [nswap, 0], [slice_size, 4])
large_w = tf.gather_nd(weights, large_i)

# update weights
indices = tf.concat((large_i, new_i), axis=0)
updates = tf.concat((large_w, new_w), axis=0)
filters = tf.scatter_nd(indices=indices, updates=updates, shape=shape)

################################################

[_abs_w, _vld_i, _vld_w, _sorted_i, _large_i, _large_w, _indices, _updates, _filters] = \
sess.run([abs_w, vld_i, vld_w, sorted_i, large_i, large_w, indices, updates, filters],  \
feed_dict={weights: x})

# print (np.shape(_abs_w))
# print (np.shape(_vld_i))
# print (np.shape(_vld_w))
# print (np.shape(_sorted_i))
# print (np.shape(_sorted_w))
print (np.shape(_large_i))
print (np.shape(_large_w))
print (np.shape(_indices))
print (np.shape(_updates))
print (np.shape(_filters))

# print (_abs_w)
# print (_vld_i)
# print (_vld_w)
# print (_sorted_i)
# print (_sorted_w)
# print (_large_i)
# print (_large_w)

################################################

