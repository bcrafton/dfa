
import numpy as np
import math

size = (10, 100)
rank = 6
sparse = 9

input_size, output_size = size
sqrt_fan_out = np.sqrt(output_size)

high = 1.0 / sqrt_fan_out
low = -high

#high = 2.
#low = -high

##################################################################

idxs = range(input_size)
combs = []

count = np.zeros(shape=input_size)
exp = np.ones(shape=input_size) * (1.0 * (rank * sparse) / input_size)

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

# assert (rank % sparse == 0)
num = int(rank / sparse)
for ii in range(2):
    tmp1 = np.random.uniform(low=low, high=high, size=(output_size, 1))
    tmp2 = np.random.uniform(low=low, high=high, size=(1, input_size))
    b = b + mask * np.dot(tmp1, tmp2)

print (b)
print (combs)
print (np.linalg.matrix_rank(b))

