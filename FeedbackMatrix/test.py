
import numpy as np
import math

size = (10, 100)
rank = 5
sparse = 1

input_size, output_size = size
sqrt_fan_out = np.sqrt(output_size)

#high = 1.0 / sqrt_fan_out
#low = -high

high = 2.
low = -high

##################################################################

idxs = range(input_size)
combs = []

for ii in range(rank):
    choice = np.random.choice(idxs, sparse, replace=True).tolist()
    combs.append(choice)
    
mask = np.zeros(shape=size)
mask = np.transpose(mask)

for ii in range(output_size):
    choices = range(len(combs))
    choice = np.random.choice(choices)
    idx = combs[choice]
    mask[ii][idx] = 1.

b = np.zeros(shape=size)
b = np.transpose(b)

for ii in range(rank):
    tmp1 = np.random.uniform(low=low, high=high, size=(output_size, 1))
    tmp2 = np.random.uniform(low=low, high=high, size=(1, input_size))
    b = b + mask * np.dot(tmp1, tmp2)

print (np.linalg.matrix_rank(mask))
print (np.linalg.matrix_rank(b))

'''
##################################################################

b = np.zeros(shape=size)
for ii in range(output_size):

    if rank > 0:
        mask = np.zeros(shape=input_size)
        mask[options[ii]] = 1
    else:
        mask = np.random.choice(idxs, sparse, replace=True)

##################################################################

for ii in range(rank):

    mask = np.zeros(shape=input_size)
    mask[options[ii]] = 1
    print (mask)

    tmp1 = np.random.uniform(low, high, size=(output_size, 1))
    tmp2 = np.random.uniform(low, high, size=(1, input_size))
    b = b + mask * (1.0 / rank) * np.dot(tmp1, tmp2)
    
##################################################################
'''
