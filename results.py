
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

cifar100_fc_bp = {'benchmark':'cifar100_fc.py', 'epochs':50, 'batch_size':64, 'alpha':[1e-4], 'l2':[0.01, 0.001], 'eps':[1e-4], 'dropout':[0.5], 'act':['relu'], 'bias':[0.1], 'dfa':[0], 'sparse':0, 'rank':0, 'init':['glorat_bp'], 'opt':['adam'], 'load':None}
cifar100_fc_dfa = {'benchmark':'cifar100_fc.py', 'epochs':50, 'batch_size':64, 'alpha':[1e-4], 'l2':[0.01, 0.001], 'eps':[1e-4], 'dropout':[0.5], 'act':['relu'], 'bias':[0.1], 'dfa':[1], 'sparse':0, 'rank':0, 'init':['glorat_dfa'], 'opt':['adam'], 'load':None}

cifar100_fc = {'benchmark':'cifar100_fc.py', 'epochs':10, 'batch_size':64, 'alpha':[1e-4, 3e-5, 1e-5], 'l2':[0.0, 0.01, 0.005, 0.001], 'eps':[1e-4, 1e-5, 1e-6], 'dropout':[0.5], 'act':['relu'], 'bias':[0.1], 'dfa':[1], 'sparse':0, 'rank':0, 'init':['zero'], 'opt':['adam'], 'load':None}

'''
cifar100_fc_bp = {'benchmark':'cifar100_fc.py', 'epochs':300, 'batch_size':64, 'alpha':[1e-4, 3e-5, 1e-5], 'l2':[0.0], 'eps':[1e-4, 1e-5, 1e-6], 'act':['tanh', 'relu'], 'bias':[0.1], 'dropout':[0.25, 0.5], 'dfa':0, 'sparse':[0], 'rank':0, 'init':'sqrt_fan_in', 'opt':['adam'], 'load':None}
cifar100_fc_dfa = {'benchmark':'cifar100_fc.py', 'epochs':300, 'batch_size':64, 'alpha':[1e-4, 3e-5, 1e-5], 'l2':[0.0], 'eps':[1e-4, 1e-5, 1e-6], 'act':['tanh', 'relu'], 'bias':[0.1], 'dropout':[0.25, 0.5], 'dfa':1, 'sparse':[0], 'rank':0, 'init':'zero', 'opt':['adam'], 'load':None}
cifar100_fc_sparse = {'benchmark':'cifar100_fc.py', 'epochs':500, 'batch_size':64, 'alpha':[1e-4, 3e-5, 1e-5], 'l2':[0.0], 'eps':[1e-4, 1e-5, 1e-6], 'act':['tanh', 'relu'], 'bias':[0.1], 'dropout':[0.25, 0.5], 'dfa':1, 'sparse':[1], 'rank':0, 'init':'zero', 'opt':['adam'], 'load':None}
'''

###############################################

# params = [imagenet_alexnet_bp, imagenet_alexnet_dfa, imagenet_alexnet_sparse]
# params = [imagenet_alexnet_dfa, imagenet_alexnet_sparse]
# params = [imagenet_alexnet_bp, imagenet_alexnet_dfa]
# params = [imagenet_alexnet_bp]
# params = [imagenet_alexnet_dfa]

params = [cifar100_fc]
params = [cifar100_fc_bp, cifar100_fc_dfa]

################################################

def get_runs():
    runs = []

    for param in params:
        perms = get_perms(param)
        runs.extend(perms)

    return runs

################################################

        
