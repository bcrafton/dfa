
import numpy as np
import os
import copy
import threading
import argparse
import matplotlib.pyplot as plt
from results import get_runs

##############################################

runs = get_runs()

##############################################

f, axes = plt.subplots(3, 4, sharex=False, sharey=False)
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

axes[0][3].set_title('acc')

num_runs = len(runs)
for ii in range(num_runs):
    param = runs[ii]

    name = '%s_%f_%f_%f_%s_%f_%f_%d_%d_%s_%s' % (param['benchmark'], param['alpha'], param['l2'], param['eps'], param['act'], param['bias'], param['dropout'], param['dfa'], param['sparse'], param['init'], param['opt'])
    if param['load']:
        name += '_transfer'
    name = name + '.npy'

    res = np.load(name).item()

    print (name)
    # print (res.keys())

    fc1 = res['fc1_std']
    dfc1 = res['dfc1']
    dfc1_bias = res['dfc1_bias']
    a1 = res['A1']
    
    fc2 = res['fc2_std']
    dfc2 = res['dfc2']
    dfc2_bias = res['dfc2_bias']
    a2 = res['A2']
    
    fc3 = res['fc3_std']
    dfc3 = res['dfc3']
    dfc3_bias = res['dfc3_bias']
    a3 = res['A3']
    
    acc = res['val_acc']

    label = '%f_%f_%d' % (param['alpha'], param['l2'], param['dfa'])
    # label = ''

    axes[0][0].plot(fc1, label=label)
    axes[1][0].plot(fc2, label=label)
    axes[2][0].plot(fc3, label=label)

    axes[0][1].plot(dfc1, label=label)
    axes[1][1].plot(dfc2, label=label)
    axes[2][1].plot(dfc3, label=label)

    axes[0][2].plot(a1, label=label)
    axes[1][2].plot(a2, label=label)
    axes[2][2].plot(a3, label=label)

    axes[0][3].plot(acc, label=label)

plt.legend()  
plt.savefig('gradients.png')


