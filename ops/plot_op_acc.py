
import numpy as np
import math
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
import argparse

# parser = argparse.ArgumentParser()
# args = parser.parse_args()

sz = 3.5
fig = plt.figure(figsize=(sz, sz))

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
itrs = range(1)
layer_shapes = (9216, 4096, 4096, 1000)

start = 0.55
accum = 0.0
delta = (0.65 - 0.55) / (len(sparses) * len(itrs))

for sparse in sparses:
    accs = []
    ops = []
    for itr in itrs:
    
        '''
        fname = args.benchmark + "/sparse%ditr%d.npy" % (sparse, itr)
        results = np.load(fname).item()
        
        acc = np.max(results['acc'])
        accs.append(acc)
        '''
        
        op = get_ops(alg='dfa', sparse=sparse, layer_shapes=layer_shapes)
        ops.append(op)

        accum += delta
        accs.append(start + accum)

        label = 'Sparse-' + str(sparse)
        plt.semilogy(accs, ops, '.', label=label, marker='.', color='#FFA500')

        ax = plt.gca()
        ax.annotate(' %d' % sparse, (accs[0], ops[0]))

#######################################

accs = []
ops = []
for itr in itrs:
    op = get_ops(alg='bp', sparse=0, layer_shapes=layer_shapes)
    ops.append(op)

    accum += delta
    accs.append(start + accum)

    label = 'BP'
    plt.semilogy(accs, ops, '.', label=label, marker='.', color='#FFA500')

    ax = plt.gca()
    ax.annotate(' BP', (accs[0], ops[0]))

#######################################
#######################################
#######################################

sparses = [1, 5, 10, 25, 100]
itrs = range(1)
layer_shapes = (4096, 2048, 2048, 100)

start = 0.45
accum = 0.0
delta = (0.55 - 0.45) / (len(sparses) * len(itrs))

for sparse in sparses:
    accs = []
    ops = []
    for itr in itrs:
    
        '''
        fname = args.benchmark + "/sparse%ditr%d.npy" % (sparse, itr)
        results = np.load(fname).item()
        
        acc = np.max(results['acc'])
        accs.append(acc)
        '''
        
        op = get_ops(alg='dfa', sparse=sparse, layer_shapes=layer_shapes)
        ops.append(op)

        accum += delta
        accs.append(start + accum)

        label = 'Sparse-' + str(sparse)
        plt.semilogy(accs, ops, label=label, marker='.', color='#2020FF')
        # plt.semilogy(accs, ops, label=label, marker='$ %d $' % (sparse), markersize=8, color='blue')

        ax = plt.gca()
        ax.annotate(' %d' % sparse, (accs[0], ops[0]))

#######################################

accs = []
ops = []
for itr in itrs:
    op = get_ops(alg='bp', sparse=0, layer_shapes=layer_shapes)
    ops.append(op)

    accum += delta
    accs.append(start + accum)

    label = 'BP'
    plt.semilogy(accs, ops, label=label, marker='.', color='#2020FF')
    
    ax = plt.gca()
    ax.annotate(' BP', (accs[0], ops[0]))

#######################################

plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['font.size'] = 10.

plt.xlabel('Accuracy', fontsize=10)
plt.xticks(fontsize=10)
ax = plt.gca()
ax.set_xticks(np.linspace(.5, .7, 5))

plt.ylabel('Ops', fontsize=10)
plt.yticks(fontsize=10)

# title = 'Ops vs Accuracy'
# plt.title(title, fontsize=10)

# plt.legend(fontsize=10)
plt.savefig('ops_vs_acc.svg', bbox_inches='tight', dpi=300)










