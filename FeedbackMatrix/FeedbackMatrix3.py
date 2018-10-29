
import numpy as np
import math

def matrix_rank(mat):
    return np.linalg.matrix_rank(B)
    
def matrix_sparsity(mat):
    a = np.sum(mat[0] != 0)
    assert(np.all(np.sum(mat != 0, axis=1) == a))
    return a

def FeedbackMatrix(size : tuple, sparse : int, rank : int):
    input_size, output_size = size
    sqrt_fan_out = np.sqrt(output_size)
    high = 1.0 / sqrt_fan_out
    low = -high

    fb = np.zeros(shape=size)
    fb = np.transpose(fb)

    choices = range(input_size)
    counts = np.zeros(input_size)
    total_connects = (1.0 * sparse * output_size)
    connects_per = (1.0 * sparse * output_size / input_size)
    
    idxs = []
    
    if sparse and rank:
        for ii in range(rank):
            choice = np.random.choice(choices, sparse, replace=False)
            idxs.append(choice)
        
        for ii in range(output_size):
            choice = np.random.choice(range(len(idxs)))
            idx = idxs[choice]
            fb[ii][idx] = 1.

    return fb

size = (10, 100)
rank = 4
sparse = 3

B = FeedbackMatrix(size, sparse, rank)

print ('rank', matrix_rank(B))
print ('sparse', matrix_sparsity(B))
