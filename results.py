
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
# use act=tanh, bias=0
# use act=relu, bias=1

# imagenet = {'benchmark':'imagenet1.py', 'epochs':100, 'batch_size':128, 'alpha':[1e-2], 'l2':[0.0], 'eps':[1.], 'dropout':[0.5], 'act':['relu'], 'bias':[1.], 'dfa':0, 'fa':[0, 1], 'sparse':0, 'rank':0, 'init':['alexnet'], 'opt':['adam'], 'load':None}
imagenet = {'benchmark':'imagenet.py', 'epochs':100, 'batch_size':128, 'alpha':[1e-2], 'l2':[0.0], 'eps':[1.], 'dropout':[0.5], 'act':['relu'], 'bias':[1.], 'dfa':0, 'fa':[0, 1], 'sparse':0, 'rank':0, 'init':['alexnet'], 'opt':['adam'], 'load':None}

################################################

params = [imagenet]

################################################

def get_runs():
    runs = []

    for param in params:
        perms = get_perms(param)
        runs.extend(perms)

    return runs

################################################

        
