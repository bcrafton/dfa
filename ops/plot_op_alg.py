
import numpy as np
import math
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
import argparse

# parser = argparse.ArgumentParser()
# args = parser.parse_args()

fig = plt.figure(figsize=(3.5, 3.5))

#######################################

def get_ops(alg, sparse, layer_shapes):
    total_ops = 0.
    
    if alg == 'bp':
        for ii in range(len(layer_shapes)-1):
            total_ops += layer_shapes[ii] * layer_shapes[ii+1]
            
    elif alg == 'dfa':
        if sparse > 0:
            for ii in range(len(layer_shapes)-1):
                total_ops += layer_shapes[ii] * sparse
        else:
            for ii in range(len(layer_shapes)-1):
                total_ops += layer_shapes[ii] * layer_shapes[-1]
            
    return total_ops

#######################################

sparses = [1, 5, 10, 25, 100, 1000]
layer_shapes = (9216, 4096, 4096, 1000)
imagenet = ('Imagenet', sparses, layer_shapes)

sparses = [1, 5, 10, 25, 100]
layer_shapes = (4096, 2048, 2048, 100)
cifar100 = ('CIFAR100', sparses, layer_shapes)

sparses = [1, 5, 10, 25, 100]
layer_shapes = (4096, 2048, 2048, 100)
cifar100 = ('CIFAR10', sparses, layer_shapes)

sparses = [1, 5, 10, 25, 100]
layer_shapes = (4096, 2048, 2048, 100)
cifar100 = ('MNIST', sparses, layer_shapes)

#######################################

for benchmark in benchmarks:
    for sparse in sparses:
        ops = []
                
        op = get_ops(alg='dfa', sparse=sparse, layer_shapes=layer_shapes)
        ops.append(op)

    op = get_ops(alg='bp', sparse=0, layer_shapes=layer_shapes)
    ops.append(op)

    label = benchmark
    plt.semilogy(sparses, ops, '.', label=label)

#######################################

plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.family'] = "sans-serif"
    
plt.xlabel('Accuracy', fontsize=10)
plt.xticks(fontsize=10)

plt.ylabel('Ops', fontsize=10)
plt.yticks(fontsize=10)

title = 'Ops vs Accuracy'
plt.title(title, fontsize=10)

plt.legend(fontsize=10)
plt.savefig('ops_vs_acc.png', bbox_inches='tight', dpi=1000)










