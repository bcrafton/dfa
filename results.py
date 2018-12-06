
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

imagenet_vgg_relu = {'benchmark':'imagenet_vgg.py', 'epochs':10, 'batch_size':32, 'alpha':[1e-4, 3e-5], 'eps':[1.], 'act':['relu'], 'bias':[1e-1, 1e-2, 1e-3], 'dropout':[0.5], 'dfa':1, 'sparse':[0], 'rank':0, 'init':['zero'], 'opt':'adam', 'load':None}
imagenet_vgg_tanh = {'benchmark':'imagenet_vgg.py', 'epochs':10, 'batch_size':32, 'alpha':[1e-2, 1e-3], 'eps':[1.], 'act':['tanh'], 'bias':[0.], 'dropout':[0.5], 'dfa':1, 'sparse':[0], 'rank':0, 'init':['zero'], 'opt':'adam', 'load':None}

###############################################

params = [imagenet_vgg_relu, imagenet_vgg_tanh]

################################################

def get_runs():
    runs = []

    for param in params:
        perms = get_perms(param)
        runs.extend(perms)

    return runs

################################################

        
