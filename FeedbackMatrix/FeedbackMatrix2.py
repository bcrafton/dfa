
import numpy as np
import math

def FeedbackMatrix(size : tuple, sparse : int, rank : int):
    input_size, output_size = size
    sqrt_fan_out = np.sqrt(output_size)
    high = 1.0 / sqrt_fan_out
    low = -high

    fb = np.zeros(shape=size)
    fb = np.transpose(fb)

    idxs = range(input_size)
    counts = np.zeros(input_size)

    total_connects = (1.0 * sparse * output_size)
    connects_per = (1.0 * sparse * output_size / input_size)
    
    for ii in range(output_size):
        if sparse:
            remaining_connects = total_connects - np.sum(counts)
            pdf = (connects_per - counts) / remaining_connects
            
            idx = np.random.choice(idxs, sparse, replace=True, p=pdf)
            fb[ii][idx] = np.random.uniform(low=low, high=high)
            counts[idx] += 1.
        
    return fb

size = (10, 100)
sparse = 1
rank = 10

B = FeedbackMatrix(size, sparse, rank)

dist = [0] * 10
for ii in range(100):
    idx = np.argmax(B[ii] != 0.0)
    if dist[idx] < 10:
        dist[idx] += 1
        
count = np.sum(dist)
print (count)



