
import numpy as np
import math

def matrix_rank(mat):
    return np.linalg.matrix_rank(B)
    
def matrix_sparsity(mat):
    a = np.sum(mat[0] != 0)
    assert (np.all(np.sum(mat != 0, axis=1) == a))
    return a
    
def full_feedback(mat):
    return (np.all(np.sum(mat != 0, axis=0) > 0))

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
        
        # pick rank sets of sparse indexes 
        for ii in range(rank):
            choice = np.random.choice(choices, sparse, replace=False)
            idxs.append(choice)

        # create our masks
        masks = []
        for ii in range(rank):
            masks.append(np.zeros(shape=(output_size, input_size)))

        for ii in range(output_size):
            choice = np.random.choice(range(len(idxs)))
            idx = idxs[choice]
            masks[choice][ii][idx] = 1.
        
        # multiply mask by random rank 1 matrix.
        for ii in range(rank):
            tmp1 = np.random.uniform(low, high, size=(output_size, 1))
            tmp2 = np.random.uniform(low, high, size=(1, input_size))
            fb = fb + masks[ii] * np.dot(tmp1, tmp2)
        
    return fb

size = (10, 100)
rank = 4
sparse = 3

B = FeedbackMatrix(size, sparse, rank)

for rank in range(10):
    for sparse in range(10):
        if rank * sparse > 10:
            B = FeedbackMatrix(size, sparse, rank)
            passed = (sparse == matrix_sparsity(B)) and (rank == matrix_rank(B)) and full_feedback(B)
            print (rank, sparse, passed)
            
            if not passed:
                print (np.sum(B != 0, axis=0))






