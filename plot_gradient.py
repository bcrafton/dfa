
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

f, axes = plt.subplots(3, 3, sharex=False, sharey=False)

num_runs = len(runs)
for ii in range(num_runs):
    param = runs[ii]

    name = '%s_%f_%f_%s_%f_%f_%d_%d_%s_%s' % (param['benchmark'], param['alpha'], param['eps'], param['act'], param['bias'], param['dropout'], param['dfa'], param['sparse'], param['init'], param['opt'])
    if param['load']:
        name += '_transfer'
    name = name + '.npy'

    res = np.load(name).item()

    fc1 = np.std(res['fc1'])
    dfc1 = res['dfc1']
    dfc1_bias = res['dfc1_bias']
    a1 = res['A1']
    
    fc2 = np.std(res['fc2'])
    dfc2 = res['dfc2']
    dfc2_bias = res['dfc2_bias']
    a2 = res['A2']
    
    fc3 = np.std(res['fc3'])
    dfc3 = res['dfc3']
    dfc3_bias = res['dfc3_bias']
    a3 = res['A3']
    
    label = '%f_%d' % (param['alpha'], param['dfa'])
    axes[0].plot(dfc1, label=label)
    axes[1].plot(dfc2, label=label)
    axes[2].plot(dfc3, label=label)
    
plt.savefig('gradients.png')
