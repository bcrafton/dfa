
import numpy as np
import math

def FeedbackMatrix(size : tuple, sparse : int, rank : int):
    input_size, output_size = size
    sqrt_fan_out = np.sqrt(output_size)

    if rank and sparse:
        assert(rank >= sparse)

    #### CREATE THE SPARSE MASK ####
    if sparse:
        mask = np.zeros(shape=(output_size, input_size))
        for ii in range(output_size):
            if rank > 0:
                idx = np.random.randint(0, rank, size=sparse)
            else:
                idx = np.random.randint(0, input_size, size=sparse)
            mask[ii][idx] = 1.0
            
        mask = np.transpose(mask)
    else:
        mask = np.ones(shape=(input_size, output_size))
    
    
    #### IF MATRIX HAS USER-SPECIFIED RANK ####
    if rank > 0:
        hi = 1.0 / sqrt_fan_out
        lo = -hi
        
        b = np.zeros(shape=(output_size, input_size))
        for ii in range(rank):
            tmp1 = np.random.uniform(lo, hi, size=(output_size, 1))
            tmp2 = np.random.uniform(lo, hi, size=(1, input_size))
            b = b + (1.0 / rank) * np.dot(tmp1, tmp2)
            
        b = np.transpose(b)
        b = b * mask
        b = b * (hi / np.average(np.absolute(b)))
        assert(np.linalg.matrix_rank(b) == rank)
    else:
        hi = 1.0 / sqrt_fan_out
        lo = -hi

        b = np.random.uniform(lo, hi, size=(input_size, output_size))
        b = b * mask
        b = b * (hi / np.average(np.absolute(b)))

    return b

size = (10, 100)
sparse = 1
rank = 10

desired_count = 97
itrs = 0
count = 0

while count < desired_count:
    B = FeedbackMatrix(size, sparse, rank)
    B = np.transpose(B)

    dist = [0] * 10
    for ii in range(100):
        idx = np.argmax(B[ii] != 0.0)
        if dist[idx] < 10:
            dist[idx] += 1
    count = np.sum(dist)

    itrs += 1
    print (itrs, count)
    




