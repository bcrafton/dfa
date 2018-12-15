
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

weights_abs = tf.abs(weights)
shape = tf.shape(weights_abs)
weights_abs = tf.reshape(weights_abs, (-1,))

where = tf.where(weights_abs > 0)
where = tf.reshape(where, (-1,))

gather = tf.gather(weights_abs, where)

small = tf.contrib.framework.argsort(gather, axis=0)
smallest = tf.gather(where, small, axis=0)

################################################

[_weights_abs, _where, _gather, _small, _smallest] = sess.run([weights_abs, where, gather, small, smallest], feed_dict={weights: x})

print (np.shape(_weights_abs))
print (np.shape(_where))
print (np.shape(_gather))
print (np.shape(_small))
print (np.shape(_smallest))

# print (_weights_abs)
# print (_gather)
# print (_small)
print (_smallest)


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
