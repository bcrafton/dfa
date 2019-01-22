
import numpy as np
import heapq
import os
import copy
import threading
import argparse
import matplotlib.pyplot as plt
from results import get_runs

##############################################

runs = get_runs()

##############################################

bp = []
dfa = []
sparse = []

f, axes = plt.subplots(3, 5, sharex=False, sharey=False)
f.set_size_inches(15, 15)

axes[0][0].set_title('fc1')
axes[1][0].set_title('fc2')
axes[2][0].set_title('fc3')

axes[0][1].set_title('dfc1')
axes[1][1].set_title('dfc2')
axes[2][1].set_title('dfc3')

axes[0][2].set_title('a1')
axes[1][2].set_title('a2')
axes[2][2].set_title('a3')

axes[0][3].set_title('ratio1')
axes[1][3].set_title('ratio2')
axes[2][3].set_title('ratio3')

axes[0][4].set_title('train_acc')
axes[1][4].set_title('test_acc')

num_runs = len(runs)
for ii in range(num_runs):
    param = runs[ii]

    name = '%s_%f_%f_%f_%s_%f_%f_%d_%d_%s_%s' % (param['benchmark'], param['alpha'], param['l2'], param['eps'], param['act'], param['bias'], param['dropout'], param['dfa'], param['sparse'], param['init'], param['opt'])
    
    if param['load']:
        name += '_transfer'
        
    name = name + '.npy'

    try:
        res = np.load(name).item()

        # print after we tried to load it.
        print (name)

        ratio1 = res['ratio1']
        fc1 = res['fc1_std']
        dfc1 = res['dfc1']
        dfc1_bias = res['dfc1_bias']
        a1 = res['A1']
        
        ratio2 = res['ratio2']
        fc2 = res['fc2_std']
        dfc2 = res['dfc2']
        dfc2_bias = res['dfc2_bias']
        a2 = res['A2']
        
        ratio3 = res['ratio3']
        fc3 = res['fc3_std']
        dfc3 = res['dfc3']
        dfc3_bias = res['dfc3_bias']
        a3 = res['A3']
        
        train_acc = res['train_acc']
        test_acc = res['test_acc']

        label = "%s %d %d %f" % (param['init'], param['dfa'], param['sparse'], param['alpha'])        

        x = (-np.max(test_acc), [ratio1, fc1, dfc1, dfc1_bias, a1, ratio2, fc2, dfc2, dfc2_bias, a2, ratio3, fc3, dfc3, dfc3_bias, a3, train_acc, test_acc, label])
        if param['dfa'] == 0:
            bp.append(x)
        elif param['dfa'] == 1 and param['sparse'] == 0:
            dfa.append(x)
        elif param['dfa'] == 1 and param['sparse'] == 1:
            sparse.append(x)
        
    except:
        pass

heapq.heapify(bp)
heapq.heapify(dfa)
heapq.heapify(sparse)

for ii in range(min(len(bp), 2)):
    _bp = bp[ii]
    ratio1, fc1, dfc1, dfc1_bias, a1, ratio2, fc2, dfc2, dfc2_bias, a2, ratio3, fc3, dfc3, dfc3_bias, a3, train_acc, test_acc, label = _bp[1]
    color = None # 'black'

    axes[0][0].plot(fc1, label=label, color=color)
    axes[1][0].plot(fc2, label=label, color=color)
    axes[2][0].plot(fc3, label=label, color=color)

    axes[0][1].plot(dfc1, label=label, color=color)
    axes[1][1].plot(dfc2, label=label, color=color)
    axes[2][1].plot(dfc3, label=label, color=color)

    axes[0][2].plot(a1, label=label, color=color)
    axes[1][2].plot(a2, label=label, color=color)
    axes[2][2].plot(a3, label=label, color=color)

    # axes[0][3].plot(ratio1, label=label, color=color)
    # axes[1][3].plot(ratio2, label=label, color=color)
    # axes[2][3].plot(ratio3, label=label, color=color)

    axes[0][4].plot(train_acc, label=label, color=color)
    axes[1][4].plot(test_acc, label=label, color=color)

for ii in range(min(len(dfa), 2)):
    _dfa = dfa[ii]
    ratio1, fc1, dfc1, dfc1_bias, a1, ratio2, fc2, dfc2, dfc2_bias, a2, ratio3, fc3, dfc3, dfc3_bias, a3, train_acc, test_acc, label = _dfa[1]
    color = None # 'red'
    
    axes[0][0].plot(fc1, label=label, color=color)
    axes[1][0].plot(fc2, label=label, color=color)
    axes[2][0].plot(fc3, label=label, color=color)

    axes[0][1].plot(dfc1, label=label, color=color)
    axes[1][1].plot(dfc2, label=label, color=color)
    axes[2][1].plot(dfc3, label=label, color=color)

    axes[0][2].plot(a1, label=label, color=color)
    axes[1][2].plot(a2, label=label, color=color)
    axes[2][2].plot(a3, label=label, color=color)

    axes[0][3].plot(ratio1, label=label, color=color)
    axes[1][3].plot(ratio2, label=label, color=color)
    axes[2][3].plot(ratio3, label=label, color=color)

    axes[0][4].plot(train_acc, label=label, color=color)
    axes[1][4].plot(test_acc, label=label, color=color)

for ii in range(min(len(sparse), 2)):
    _sparse = sparse[ii]
    ratio1, fc1, dfc1, dfc1_bias, a1, ratio2, fc2, dfc2, dfc2_bias, a2, ratio3, fc3, dfc3, dfc3_bias, a3, train_acc, test_acc, label = _sparse[1]
    color = None # 'blue'

    axes[0][0].plot(fc1, label=label, color=color)
    axes[1][0].plot(fc2, label=label, color=color)
    axes[2][0].plot(fc3, label=label, color=color)    

    axes[0][1].plot(dfc1, label=label, color=color)
    axes[1][1].plot(dfc2, label=label, color=color)
    axes[2][1].plot(dfc3, label=label, color=color)

    axes[0][2].plot(a1, label=label, color=color)
    axes[1][2].plot(a2, label=label, color=color)
    axes[2][2].plot(a3, label=label, color=color)

    axes[0][3].plot(ratio1, label=label, color=color)
    axes[1][3].plot(ratio2, label=label, color=color)
    axes[2][3].plot(ratio3, label=label, color=color)

    axes[0][4].plot(train_acc, label=label, color=color)
    axes[1][4].plot(test_acc, label=label, color=color)

axes[0][3].legend()
axes[1][3].legend()
axes[2][3].legend()

axes[0][4].legend()  
f.savefig('gradients.png')














