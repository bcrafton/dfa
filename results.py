
import numpy as np
import os
import copy
import threading
import argparse

################################################

def get_perms(param):
    params = [param]
    
    for key in param.keys():
        val = param[key]
        if type(val) == list:
            new_params = []
            for ii in range(len(val)):
                for jj in range(len(params)):
                    new_param = copy.copy(params[jj])
                    new_param[key] = val[ii]
                    new_params.append(new_param)
                    
            params = new_params
            
    return params

################################################

# wanted to run this with tanh as well.
cifar100_fc_bp = {'benchmark':'cifar100_fc.py', 'epochs':100, 'batch_size':64, 'alpha':[1e-4], 'l2':[1e-3, 1e-4], 'eps':[1e-4, 1e-5], 'dropout':[0.5], 'act':['tanh', 'relu'], 'bias':[0.1, 1.0], 'dfa':[0], 'sparse':0, 'rank':0, 'init':['glorat_bp'], 'opt':['adam'], 'load':None}
cifar100_fc_dfa = {'benchmark':'cifar100_fc.py', 'epochs':100, 'batch_size':64, 'alpha':[1e-4], 'l2':[1e-3, 5e-3, 1e-4], 'eps':[1e-4, 1e-5], 'dropout':[0.5], 'act':['tanh', 'relu'], 'bias':[0.1, 1.0], 'dfa':[1], 'sparse':0, 'rank':0, 'init':['glorat_dfa'], 'opt':['adam'], 'load':None}

cifar100_fc_bp = {'benchmark':'cifar100_fc_4096.py', 'epochs':300, 'batch_size':64, 'alpha':[1e-4], 'l2':[1e-4], 'eps':[1e-4], 'dropout':[0.5], 'act':['relu'], 'bias':[0.1], 'dfa':[0], 'sparse':0, 'rank':0, 'init':['glorat_bp', 'glorat_dfa'], 'opt':['adam'], 'load':None}
cifar100_fc_dfa = {'benchmark':'cifar100_fc_4096.py', 'epochs':300, 'batch_size':64, 'alpha':[1e-4], 'l2':[1e-4], 'eps':[1e-4], 'dropout':[0.5], 'act':['relu'], 'bias':[0.1], 'dfa':[1], 'sparse':0, 'rank':0, 'init':['glorat_dfa'], 'opt':['adam'], 'load':None}
# cifar100_fc_sparse1 = {'benchmark':'cifar100_fc_4096.py', 'epochs':500, 'batch_size':64, 'alpha':[5e-5, 7e-5], 'l2':[1e-4], 'eps':[1e-4], 'dropout':[0.5], 'act':['relu'], 'bias':[0.1], 'dfa':[1], 'sparse':1, 'rank':0, 'init':['glorat_dfa'], 'opt':['adam'], 'load':None}
cifar100_fc_sparse1 = {'benchmark':'cifar100_fc_4096.py', 'epochs':500, 'batch_size':64, 'alpha':[1e-4], 'l2':[0.], 'eps':[1e-4], 'dropout':[0.5], 'act':['relu'], 'bias':[1.0], 'dfa':[1], 'sparse':1, 'rank':0, 'init':['zero'], 'opt':['adam'], 'load':None}
cifar100_fc_sparse2 = {'benchmark':'cifar100_fc_4096.py', 'epochs':500, 'batch_size':64, 'alpha':[1e-4, 2e-4], 'l2':[1e-4], 'eps':[1e-4], 'dropout':[0.5], 'act':['relu'], 'bias':[0.1], 'dfa':[1], 'sparse':1, 'rank':0, 'init':['glorat_dfa'], 'opt':['adam'], 'load':None}


###############################################

# params = [cifar100_fc_bp]
# params = [cifar100_fc_dfa]
params = [cifar100_fc_sparse1]
# params = [cifar100_fc_sparse2]
# params = [cifar100_fc_bp, cifar100_fc_dfa, cifar100_fc_sparse1, cifar100_fc_sparse2]

################################################

def get_runs():
    runs = []

    for param in params:
        perms = get_perms(param)
        runs.extend(perms)

    return runs

################################################

        
