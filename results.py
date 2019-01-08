
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

cifar10_conv_bp = {'benchmark':'cifar10_conv.py', 'epochs':100, 'batch_size':64, 'alpha':[1e-4, 1e-5], 'eps':[1e-4, 1e-5, 1e-6], 'l2':[0.], 'rate':[0.25, 0.5], 'swap':[0.25, 0.5], 'dropout':[0.5], 'act':['relu'], 'bias':[1.], 'dfa':0, 'sparse':0, 'rank':0, 'init':['sqrt_fan_in'], 'opt':['adam'], 'load':None}

cifar10_conv_dfa = {'benchmark':'cifar10_conv.py', 'epochs':100, 'batch_size':64, 'alpha':[1e-4, 1e-5], 'eps':[1e-4, 1e-5, 1e-6], 'l2':[0.], 'rate':[0.25, 0.5], 'swap':[0.25, 0.5], 'dropout':[0.5], 'act':['relu'], 'bias':[1.], 'dfa':1, 'sparse':0, 'rank':0, 'init':['sqrt_fan_in'], 'opt':['adam'], 'load':None}

###############################################

params = [cifar10_conv_bp, cifar10_conv_dfa]

################################################

def get_runs():
    runs = []

    for param in params:
        perms = get_perms(param)
        runs.extend(perms)

    return runs

################################################

        
