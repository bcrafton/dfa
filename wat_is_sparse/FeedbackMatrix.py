
import numpy as np
import math
import matplotlib.pyplot as plt
import argparse

#######################################

parser = argparse.ArgumentParser()
parser.add_argument('--axis', type=int, default=0)
args = parser.parse_args()

#######################################

np.random.seed(0)

def matrix_rank(mat):
    return np.linalg.matrix_rank(B)
    
def matrix_sparsity(mat):
    # we want to make sure all the [10]s have some sparisty.
    # so we transpose the matrix from [10, 100] -> [100, 10] so that way we can look at the first [10]
    # summing along the 0 axis, sums along the 10s ... so we get a [100] vector sum 
    
    a = np.sum(mat.T[0] != 0)
    assert (np.all(np.sum(mat != 0, axis=0) == a))
    return a
    
def full_feedback(mat):
    # sum along the opposite axis here.
    
    return (np.all(np.sum(mat != 0, axis=1) > 0))

def FeedbackMatrix(size : tuple, sparse : int, rank : int):
    input_size, output_size = size

    if sparse:
        sqrt_fan_out = np.sqrt(1.0 * output_size / input_size * sparse)
    else:
        sqrt_fan_out = np.sqrt(output_size)

    high = 1.0 / sqrt_fan_out
    low = -high

    fb = np.zeros(shape=size)
    fb = np.transpose(fb)

    choices = range(input_size)
    counts = np.zeros(input_size)
    total_connects = (1.0 * sparse * rank)
    connects_per = (1.0 * sparse * rank / input_size)
    
    idxs = []
    
    if sparse and rank:
        assert(sparse * rank >= input_size)
        
        # pick rank sets of sparse indexes 
        for ii in range(rank):
            remaining_connects = total_connects - np.sum(counts)
            pdf = (connects_per - counts) / remaining_connects
            pdf = np.clip(pdf, 1e-6, 1.0)
            pdf = pdf / np.sum(pdf)
            
            choice = np.random.choice(choices, sparse, replace=False, p=pdf)
            counts[choice] += 1.
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
            
        # rank fix
        fb = fb * (high / np.std(fb))
        fb = fb.T
        
    elif sparse:
        mask = np.zeros(shape=(output_size, input_size))
        for ii in range(output_size):
            idx = np.random.choice(choices, size=sparse, replace=False)
            mask[ii][idx] = 1.0
        
        mask = mask.T
        fb = np.random.uniform(low, high, size=(input_size, output_size))
        fb = fb * mask
        
    elif rank:
        fb = np.zeros(shape=(input_size, output_size))
        for ii in range(rank):
            tmp1 = np.random.uniform(low, high, size=(input_size, 1))
            tmp2 = np.random.uniform(low, high, size=(1, output_size))
            fb = fb + np.dot(tmp1, tmp2)
        # rank fix
        fb = fb * (high / np.std(fb))

    else:
        fb = np.random.uniform(low, high, size=(input_size, output_size))

    return fb

size = (10, 25)
rank = 0
sparse = 1

B = np.absolute(FeedbackMatrix(size, sparse, rank)).T

####################################

plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['font.size'] = 10.

f, ax = plt.subplots()
# f.subplots_adjust(hspace=0)
f.set_size_inches(3.5, 3.5)

im = ax.imshow(B, cmap="gray_r")

if args.axis:
    ax.set_xticks([0, 10])
    ax.set_yticks([0, 25])
else:
    ax.set_xticks([])
    ax.set_yticks([])


cbar = f.colorbar(im, ax=ax)
# ticklabs = cbar.ax.get_yticklabels()
# cbar.ax.set_yticklabels(ticklabs, fontsize=10)

# plt.savefig('B.png', dpi=1000, bbox_inches='tight')
plt.savefig('B.png', dpi=1000)



