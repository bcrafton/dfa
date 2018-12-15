
import numpy as np

shape = (10, 10)
x = np.zeros(shape=shape)
x = np.reshape(x, (-1))
x[np.random.randint(0, 100, 25)] = np.random.rand(25)

y = np.where(x == 0)[0].tolist()
x[y] = np.inf
y = np.argsort(x, axis=None)[0:10]
x[y] = 0.
y = np.where(x == np.inf)[0].tolist()
x[y] = 0.

x = np.reshape(x, shape)

print (x)
