
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

results = {}

num_runs = len(runs)
for ii in range(num_runs):
    param = runs[ii]

    # figure out the name of the param
    name = '%s_%f_%f_%s_%f_%f_%d_%d_%s_%s' % (param['benchmark'], param['alpha'], param['eps'], param['act'], param['bias'], param['dropout'], param['dfa'], param['sparse'], param['init'], param['opt'])
    if param['load']:
        name += '_transfer'
    name = name + '.npy'

    # load the results
    res = np.load(name).item()

    dfc1 = res['dfc1']
    dfc1_bias = res['dfc1_bias']
    
    dfc2 = res['dfc2']
    dfc2_bias = res['dfc2_bias']
    
    dfc3 = res['dfc3']
    dfc3_bias = res['dfc3_bias']

    plt.plot(dfc1)

plt.savefig('plot_gradients.png')
