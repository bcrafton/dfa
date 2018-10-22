
import numpy as np
import math
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
import argparse

# parser = argparse.ArgumentParser()
# args = parser.parse_args()

fig = plt.figure(figsize=(10, 10))

#######################################

def get_ops(alg, sparse, layer_shapes):
    total_ops = 0.
    
    if alg == 'bp':
        for ii in range(len(layer_shapes)-1):
            total_ops += layer_shapes[ii] * layer_shapes[ii+1]
            
    elif alg == 'dfa':
        if sparse > 0:
            for ii in range(len(layer_shapes)-1):
                total_ops += sparse * layer_shapes[-1]
        else:
            for ii in range(len(layer_shapes)-1):
                total_ops += layer_shapes[ii] * layer_shapes[-1]
            
    return total_ops

#######################################

# benchmarks = ['cifar100', 'imagenet']
sparses = [0, 1, 5, 10, 25, 100]
itrs = range(1)
layer_shapes = (9216, 4096, 4096, 1000)

accs = []
ops = []

for sparse in sparses:
    for itr in itrs:
    
        '''
        fname = args.benchmark + "/sparse%ditr%d.npy" % (sparse, itr)
        results = np.load(fname).item()
        
        acc = np.max(results['acc'])
        accs.append(acc)
        '''
        
        op = get_ops(alg='dfa', sparse=sparse, layer_shapes=layer_shapes)
        ops.append(op)

accs = np.linspace(0.55, 0.65, len(sparses))

#######################################

plt.scatter(accs, ops, s=30)
    
plt.xlabel('Accuracy', fontsize=18)
plt.xticks(fontsize=14)

plt.ylabel('Ops', fontsize=18)
plt.yticks(fontsize=14)

title = 'Ops vs Accuracy'
plt.title(title, fontsize=18)

plt.savefig('ops_vs_acc.png')










