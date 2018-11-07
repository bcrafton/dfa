
import numpy as np
import os
import copy
import threading
import argparse
from results import get_runs

##############################################

runs = get_runs()

##############################################

results = {}

num_runs = len(runs)
for ii in range(num_runs):
    param = runs[ii]

    # figure out the name of the param
    name = '%s_%f_%d_%d_%s_%s' % (param['benchmark'], param['alpha'], param['dfa'], param['sparse'], param['init'], param['opt'])
    if param['load']:
        name += '_transfer'
    name = name + '.npy'

    # load the results
    res = np.load(name).item()
    
    if param['load']:
        transfer = 1
    else:
        transfer = 0
    
    key = (param['benchmark'], param['dfa'], param['sparse'], transfer)
    val = max(res['test_acc'])
    
    if key in results.keys():
        # use an if instead of max because we gonna want to save the winner run information
        if results[key][0] < val:
            results[key] = (val, param['benchmark'], param['alpha'], param['dfa'], param['sparse'], param['init'], param['opt'])
    else:
        results[key] = (val, param['benchmark'], param['alpha'], param['dfa'], param['sparse'], param['init'], param['opt'])
            
for key in sorted(results.keys()):   
    print (key, results[key])





        
