
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

mnist_fc_lel = {'benchmark':'mnist_fc.py', 'epochs':300, 'batch_size':32, 'alpha':[1e-4, 3e-5, 1e-5], 'eps':[1e-4, 1e-5, 1e-6], 'act':['tanh', 'relu'], 'dropout':[0.25, 0.5], 'alg':['lel'], 'sparse':[0, 1], 'rank':0, 'init':'zero', 'opt':'gd', 'load':[None]}
mnist_fc_sparse = {'benchmark':'mnist_fc.py', 'epochs':300, 'batch_size':32, 'alpha':[1e-4, 3e-5, 1e-5], 'eps':[1e-4, 1e-5, 1e-6], 'act':['tanh', 'relu'], 'dropout':[0.25, 0.5], 'alg':['lel'], 'sparse':[0, 1], 'rank':0, 'init':'zero', 'opt':'gd', 'load':[None]}

cifar10_fc_lel = {'benchmark':'cifar10_fc.py', 'epochs':300, 'batch_size':64, 'alpha':[1e-4, 3e-5, 1e-5], 'eps':[1e-4, 1e-5, 1e-6], 'act':['tanh', 'relu'], 'dropout':[0.25, 0.5], 'alg':['lel'], 'sparse':[0], 'rank':0, 'init':'zero', 'opt':['adam'], 'load':[None]}
cifar10_fc_lel_sparse = {'benchmark':'cifar10_fc.py', 'epochs':300, 'batch_size':64, 'alpha':[1e-4, 3e-5, 1e-5], 'eps':[1e-4, 1e-5, 1e-6], 'act':['tanh', 'relu'], 'dropout':[0.25, 0.5], 'alg':['lel'], 'sparse':[1], 'rank':0, 'init':'zero', 'opt':['adam'], 'load':[None]}

cifar100_fc_lel = {'benchmark':'cifar100_fc.py', 'epochs':300, 'batch_size':64, 'alpha':[1e-4, 3e-5, 1e-5], 'eps':[1e-4, 1e-5, 1e-6], 'act':['tanh', 'relu'], 'dropout':[0.25, 0.5], 'alg':['lel'], 'sparse':[0], 'rank':0, 'init':'zero', 'opt':['adam'], 'load':[None]}
cifar100_fc_lel_sparse = {'benchmark':'cifar100_fc.py', 'epochs':300, 'batch_size':64, 'alpha':[1e-4, 3e-5, 1e-5], 'eps':[1e-4, 1e-5, 1e-6], 'act':['tanh', 'relu'], 'dropout':[0.25, 0.5], 'alg':['lel'], 'sparse':[1], 'rank':0, 'init':'zero', 'opt':['adam'], 'load':[None]}

################################################

mnist_conv_lel = {'benchmark':'mnist_conv.py', 'epochs':300, 'batch_size':64, 'alpha':[0.01, 0.005], 'eps':[1e-4, 1e-5, 1e-6], 'act':['tanh', 'relu'], 'dropout':[0.25, 0.5], 'alg':['lel'], 'sparse':[0, 1], 'rank':0, 'init':'zero', 'opt':'gd', 'load':[None]}
mnist_conv_lel_sparse = {'benchmark':'mnist_conv.py', 'epochs':300, 'batch_size':64, 'alpha':[0.01, 0.005], 'eps':[1e-4, 1e-5, 1e-6], 'act':['tanh', 'relu'], 'dropout':[0.25, 0.5], 'alg':['lel'], 'sparse':[0, 1], 'rank':0, 'init':'zero', 'opt':'gd', 'load':[None]}

cifar10_conv_lel = {'benchmark':'cifar10_conv.py', 'epochs':300, 'batch_size':64, 'alpha':[1e-4, 3e-5, 1e-5], 'eps':[1e-4, 1e-5, 1e-6], 'act':['tanh', 'relu'], 'dropout':[0.25, 0.5], 'alg':['lel'], 'sparse':[0], 'rank':0, 'init':'zero', 'opt':['adam'], 'load':[None]}
cifar10_conv_lel_sparse = {'benchmark':'cifar10_conv.py', 'epochs':300, 'batch_size':64, 'alpha':[1e-4, 3e-5, 1e-5], 'eps':[1e-4, 1e-5, 1e-6], 'act':['tanh', 'relu'], 'dropout':[0.25, 0.5], 'alg':['lel'], 'sparse':[1], 'rank':0, 'init':'zero', 'opt':['adam'], 'load':[None]}

cifar100_conv_lel = {'benchmark':'cifar100_conv.py', 'epochs':300, 'batch_size':64, 'alpha':[1e-4, 3e-5, 1e-5], 'eps':[1e-4, 1e-5, 1e-6], 'act':['tanh', 'relu'], 'dropout':[0.25, 0.5], 'alg':['lel'], 'sparse':[0], 'rank':0, 'init':'zero', 'opt':['adam'], 'load':[None]}
cifar100_conv_lel_sparse = {'benchmark':'cifar100_conv.py', 'epochs':300, 'batch_size':64, 'alpha':[1e-4, 3e-5, 1e-5], 'eps':[1e-4, 1e-5, 1e-6], 'act':['tanh', 'relu'], 'dropout':[0.25, 0.5], 'alg':['lel'], 'sparse':[1], 'rank':0, 'init':'zero', 'opt':['adam'], 'load':[None]}

################################################

params = [mnist_fc_lel, cifar10_fc_lel, cifar100_fc_lel, mnist_conv_lel, cifar10_conv_lel, cifar100_conv_lel]

################################################

def get_runs():
    runs = []

    for param in params:
        perms = get_perms(param)
        runs.extend(perms)

    return runs

################################################

        
