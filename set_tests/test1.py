
import numpy as np

shape = (10, 10)
x = np.zeros(shape=shape)
x = np.reshape(x, (-1))
x[np.random.randint(0, 100, 25)] = np.random.rand(25)

a = np.absolute(x)
valid_idx = np.where(a > 0)[0]
out = valid_idx[a[valid_idx].argsort()[0:10]]

print (a[valid_idx])
print (a[valid_idx].argsort())
print (a[valid_idx].argsort()[0:10])

# print (x)
# print (out)
# print (x[out])
