
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

imagenet_alexnet_bp = {'benchmark':'alexnet_fc.py', 'epochs':100, 'batch_size':128, 'alpha':[1e-2, 3e-3, 1e-3, 3e-4], 'l2':[0.0], 'eps':[1.], 'dropout':[0.5], 'act':['relu'], 'bias':[1.], 'dfa':0, 'sparse':0, 'rank':0, 'init':['alexnet'], 'opt':['adam'], 'load':None}
imagenet_alexnet_dfa = {'benchmark':'alexnet_fc.py', 'epochs':100, 'batch_size':128, 'alpha':[1e-2, 3e-3, 1e-3, 3e-4], 'l2':[0.0], 'eps':[1.], 'dropout':[0.5], 'act':['relu'], 'bias':[1.], 'dfa':1, 'sparse':0, 'rank':0, 'init':['zero'], 'opt':['adam'], 'load':None}
# imagenet_alexnet_sparse = {'benchmark':'alexnet_fc.py', 'epochs':100, 'batch_size':128, 'alpha':[3e-3, 1e-3, 3e-4, 1e-4], 'l2':[0.0], 'eps':[1.], 'dropout':[0.1], 'act':['relu'], 'bias':[1.], 'dfa':1, 'sparse':1, 'rank':0, 'init':['zero'], 'opt':'adam', 'load':None}
imagenet_alexnet_sparse = {'benchmark':'alexnet_fc.py', 'epochs':100, 'batch_size':128, 'alpha':[1e-3], 'l2':[0.0], 'eps':[1.], 'dropout':[0.05, 0.1, 0.25, 0.33], 'act':['relu'], 'bias':[1.], 'dfa':1, 'sparse':1, 'rank':0, 'init':['zero'], 'opt':'adam', 'load':None}

imagenet_vgg_bp = {'benchmark':'vgg_fc.py', 'epochs':100, 'batch_size':32, 'alpha':[0.01, 0.003, 0.001, 0.0003], 'l2':[0.], 'eps':[1.], 'act':['relu'], 'bias':[1.], 'dropout':[0.5], 'dfa':0, 'sparse':0, 'rank':0, 'init':['alexnet'], 'opt':'adam', 'load':None}
imagenet_vgg_dfa = {'benchmark':'vgg_fc.py', 'epochs':100, 'batch_size':32, 'alpha':[0.01, 0.003, 0.001, 0.0003], 'l2':[0.], 'eps':[1.], 'act':['relu'], 'bias':[1.], 'dropout':[0.5], 'dfa':1, 'sparse':0, 'rank':0, 'init':['zero'], 'opt':'adam', 'load':None}


imagenet_vgg_bp = {'benchmark':'vgg_fc.py', 'epochs':100, 'batch_size':32, 'alpha':[0.001, 0.0003], 'l2':[0.], 'eps':[1.], 'act':['relu'], 'bias':[1.], 'dropout':[0.5], 'dfa':0, 'sparse':0, 'rank':0, 'init':['alexnet'], 'opt':'adam', 'load':None}
imagenet_vgg_dfa = {'benchmark':'vgg_fc.py', 'epochs':100, 'batch_size':32, 'alpha':[0.0003, 0.0001], 'l2':[0.], 'eps':[1.], 'act':['relu'], 'bias':[1.], 'dropout':[0.5], 'dfa':1, 'sparse':0, 'rank':0, 'init':['zero'], 'opt':'adam', 'load':None}

imagenet_vgg_sparse1 = {'benchmark':'vgg_fc.py', 'epochs':100, 'batch_size':64, 'alpha':[0.05], 'l2':[0.], 'eps':[1.], 'act':['tanh'], 'bias':[0.], 'dropout':[0.5], 'dfa':1, 'sparse':1, 'rank':0, 'init':['zero'], 'opt':'adam', 'load':None}
imagenet_vgg_sparse2 = {'benchmark':'vgg_fc.py', 'epochs':100, 'batch_size':32, 'alpha':[0.0001, 0.0003], 'l2':[0.], 'eps':[1.], 'act':['relu'], 'bias':[1.], 'dropout':[0.05, 0.15], 'dfa':1, 'sparse':1, 'rank':0, 'init':['zero'], 'opt':'adam', 'load':None}
imagenet_vgg_sparse3 = {'benchmark':'vgg_fc.py', 'epochs':100, 'batch_size':32, 'alpha':[0.0001], 'l2':[0.], 'eps':[1.], 'act':['relu'], 'bias':[1.], 'dropout':[0.25], 'dfa':1, 'sparse':1, 'rank':0, 'init':['zero'], 'opt':'adam', 'load':None}

################################################

#dfa3
alexnet_bp = {'benchmark':'alexnet_fc.py', 'epochs':100, 'batch_size':128, 'alpha':[1e-2], 'l2':[0.0], 'eps':[1.], 'dropout':[0.5], 'act':['relu'], 'bias':[1.], 'dfa':0, 'sparse':0, 'rank':0, 'init':['alexnet'], 'opt':['adam'], 'load':None}
# dfa3
alexnet_dfa = {'benchmark':'alexnet_fc.py', 'epochs':100, 'batch_size':128, 'alpha':[1e-3], 'l2':[0.0], 'eps':[1.], 'dropout':[0.5], 'act':['relu'], 'bias':[1.], 'dfa':1, 'sparse':0, 'rank':0, 'init':['zero'], 'opt':['adam'], 'load':None}
# dfa5
alexnet_sparse = {'benchmark':'alexnet_fc.py', 'epochs':100, 'batch_size':128, 'alpha':[1e-3], 'l2':[0.0], 'eps':[1.], 'dropout':[0.5], 'act':['relu'], 'bias':[1.], 'dfa':1, 'sparse':1, 'rank':0, 'init':['zero'], 'opt':'adam', 'load':None}

#dfa5
vgg_bp = {'benchmark':'vgg_fc.py', 'epochs':100, 'batch_size':32, 'alpha':[0.001], 'l2':[0.], 'eps':[1.], 'act':['relu'], 'bias':[1.], 'dropout':[0.5], 'dfa':0, 'sparse':0, 'rank':0, 'init':['alexnet'], 'opt':'adam', 'load':None}
#dfa5
vgg_dfa = {'benchmark':'vgg_fc.py', 'epochs':100, 'batch_size':32, 'alpha':[0.0003], 'l2':[0.], 'eps':[1.], 'act':['relu'], 'bias':[1.], 'dropout':[0.5], 'dfa':1, 'sparse':0, 'rank':0, 'init':['zero'], 'opt':'adam', 'load':None}
#dfa3
vgg_sparse = {'benchmark':'vgg_fc.py', 'epochs':100, 'batch_size':32, 'alpha':[0.0003], 'l2':[0.], 'eps':[1.], 'act':['relu'], 'bias':[1.], 'dropout':[0.0], 'dfa':1, 'sparse':1, 'rank':0, 'init':['zero'], 'opt':'adam', 'load':None}

################################################

# params = [imagenet_vgg_bp]
# params = [imagenet_vgg_dfa]
# params = [imagenet_vgg_sparse]
# params = [imagenet_vgg_bp, imagenet_vgg_dfa, imagenet_vgg_sparse]
# params = [imagenet_vgg_dfa, imagenet_vgg_sparse]
# params = [imagenet_vgg_sparse1, imagenet_vgg_sparse2, imagenet_vgg_sparse3]
# params = [imagenet_vgg_sparse2]
# params = [imagenet_vgg_bp, imagenet_vgg_dfa]
# params = [imagenet_alexnet_sparse]

# params = [imagenet_vgg_sparse1, imagenet_vgg_sparse2, imagenet_vgg_sparse3, imagenet_vgg_bp, imagenet_vgg_dfa]
# params = [imagenet_vgg_sparse2, imagenet_vgg_bp, imagenet_vgg_dfa, imagenet_alexnet_sparse]

params = [alexnet_bp, alexnet_dfa, alexnet_sparse, vgg_bp, vgg_dfa, vgg_sparse]
# params = [vgg_bp, vgg_dfa, vgg_sparse]

################################################

def get_runs():
    runs = []

    for param in params:
        perms = get_perms(param)
        runs.extend(perms)

    return runs

################################################

        
