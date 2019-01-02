
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
    
    results = {}
    
    results['dfc1'] =      dfc1
    results['dfc1_bias'] = dfc1_bias
    results['dfc2'] =      dfc2
    results['dfc2_bias'] = dfc2_bias
    results['dfc3'] =      dfc3
    results['dfc3_bias'] = dfc3_bias

    name = 'gradients_' + name
    np.save(name, results)
