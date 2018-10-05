
import numpy as np
import math

size = (5, 20)
rank = 2
sparse = 2

input_size, output_size = size
sqrt_fan_out = np.sqrt(output_size)

#high = 1.0 / sqrt_fan_out
#low = -high

high = 2.
low = -high

##################################################################

idxs = range(input_size)
combs = []

count = np.zeros(shape=input_size)
exp = np.ones(shape=input_size) * (1.0 * (rank * sparse) / input_size)
total = rank * sparse

for ii in range(rank):
    pdf = 1.0 * (exp - count)
    pdf = np.clip(pdf, 0.0, 1.0)
    pdf = pdf / np.sum(pdf)
    # print (pdf, np.sum(pdf))

    choice = np.random.choice(idxs, sparse, replace=False, p=pdf).tolist()
    combs.append(choice)
    
    count[choice] += 1

mask = np.zeros(shape=size)
mask = np.transpose(mask)

for ii in range(output_size):
    choices = range(len(combs))
    choice = np.random.choice(choices, replace=False)
    idx = combs[choice]
    mask[ii][idx] = 1.

b = np.zeros(shape=size)
b = np.transpose(b)

for ii in range(rank):
    tmp1 = np.random.uniform(low=low, high=high, size=(output_size, 1))
    tmp2 = np.random.uniform(low=low, high=high, size=(1, input_size))
    b = b + mask * np.dot(tmp1, tmp2)

# b = np.transpose(b)
# print (np.transpose(np.where(b != 0)))
print (b)
print (combs)
print (np.linalg.matrix_rank(b))

x = np.zeros(shape=(10, 10))
mask1 = np.zeros(shape=(10, 10))
mask1[:, 1] = 1
mask1[:, 2] = 1
mask1[:, 3] = 1
mask2 = np.zeros(shape=(10, 10))
mask2[:, 0] = 1
mask2[:, 4] = 1
mask2[:, 7] = 1

for ii in range(10):
    if np.random.choice([0, 1]):
        mask1[ii] *= 1
        mask2[ii] *= 0
    else:
        mask1[ii] *= 0
        mask2[ii] *= 1

# print (mask1)
# print (mask2)

mask = mask1 + mask2

for ii in range(1):
    tmp1 = np.random.uniform(low=low, high=high, size=(10, 1))
    tmp2 = np.random.uniform(low=low, high=high, size=(1, 10))
    x = x + mask * np.dot(tmp1, tmp2)

print (x)
print (np.linalg.matrix_rank(x))

### OH WOW rank = sparse * # vectors used.
# that is great!!!


