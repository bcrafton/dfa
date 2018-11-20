
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

# mnist_fc_bp = {'benchmark':'mnist_fc.py', 'epochs':300, 'batch_size':32, 'alpha':[0.05, 0.03], 'dfa':0, 'sparse':0, 'rank':0, 'init':'sqrt_fan_in', 'opt':'gd', 'load':None}
# mnist_fc_dfa = {'benchmark':'mnist_fc.py', 'epochs':300, 'batch_size':32, 'alpha':[0.1, 0.075, 0.05, 0.03], 'dfa':1, 'sparse':[0, 1], 'rank':0, 'init':'zero', 'opt':'gd', 'load':None}

'''
cifar10_fc_bp = {'benchmark':'cifar10_fc.py', 'epochs':300, 'batch_size':64, 'alpha':[0.02, 0.01, 0.005, 0.001], 'dfa':0, 'sparse':0, 'rank':0, 'init':'sqrt_fan_in', 'opt':['gd', 'adam'], 'load':None}
cifar10_fc_dfa = {'benchmark':'cifar10_fc.py', 'epochs':300, 'batch_size':64, 'alpha':[0.02, 0.01, 0.005, 0.001], 'dfa':1, 'sparse':[0, 1], 'rank':0, 'init':'zero', 'opt':['gd', 'adam'], 'load':None}

cifar100_fc_bp = {'benchmark':'cifar100_fc.py', 'epochs':300, 'batch_size':64, 'alpha':[0.02, 0.01, 0.005, 0.001], 'dfa':0, 'sparse':0, 'rank':0, 'init':'sqrt_fan_in', 'opt':['gd', 'adam'], 'load':None}
cifar100_fc_dfa = {'benchmark':'cifar100_fc.py', 'epochs':300, 'batch_size':64, 'alpha':[0.02, 0.01, 0.005, 0.001], 'dfa':1, 'sparse':[0, 1], 'rank':0, 'init':'zero', 'opt':['gd', 'adam'], 'load':None}
'''

cifar10_fc_bp = {'benchmark':'cifar10_fc.py', 'epochs':500, 'batch_size':64, 'alpha':[1e-4, 3e-5, 1e-5], 'eps':[1e-4, 1e-5, 1e-6], 'dfa':0, 'sparse':0, 'rank':0, 'init':'sqrt_fan_in', 'opt':['adam'], 'load':None}
cifar10_fc_dfa = {'benchmark':'cifar10_fc.py', 'epochs':1000, 'batch_size':64, 'alpha':[1e-4, 3e-5, 1e-5], 'eps':[1e-4, 1e-5, 1e-6], 'dfa':1, 'sparse':[0], 'rank':0, 'init':'zero', 'opt':['adam'], 'load':None}

cifar10_fc_bp1 = {'benchmark':'cifar10_fc.py', 'epochs':500, 'batch_size':64,   'alpha':[5e-3, 1e-2], 'eps':[1e-1, 1.], 'dfa':0, 'sparse':0, 'rank':0, 'init':'sqrt_fan_in', 'opt':['adam'], 'load':None}
cifar10_fc_dfa1 = {'benchmark':'cifar10_fc.py', 'epochs':1000, 'batch_size':64, 'alpha':[5e-3, 1e-2], 'eps':[1e-1, 1.], 'dfa':1, 'sparse':[0], 'rank':0, 'init':'zero', 'opt':['adam'], 'load':None}

'''
cifar100_fc_bp = {'benchmark':'cifar100_fc.py', 'epochs':100, 'batch_size':64, 'alpha':[0.0003, 0.0001, 0.00003], 'eps':[1e-4, 1e-6, 1e-8], 'dfa':0, 'sparse':0, 'rank':0, 'init':'sqrt_fan_in', 'opt':['adam'], 'load':None}
cifar100_fc_dfa = {'benchmark':'cifar100_fc.py', 'epochs':100, 'batch_size':64, 'alpha':[0.0003, 0.0001, 0.00003], 'eps':[1e-4, 1e-6, 1e-8], 'dfa':1, 'sparse':[0, 1], 'rank':0, 'init':'zero', 'opt':['adam'], 'load':None}
'''

# cifar10_fc_dfa = {'benchmark':'cifar10_fc.py', 'epochs':500, 'batch_size':64, 'alpha':[0.02, 0.01], 'dfa':1, 'sparse':[0], 'rank':0, 'init':'zero', 'opt':['gd'], 'load':None}

################################################

'''
mnist_conv_bp = {'benchmark':'mnist_conv.py', 'epochs':300, 'batch_size':64, 'alpha':[0.01, 0.005], 'dfa':0, 'sparse':0, 'rank':0, 'init':'sqrt_fan_in', 'opt':'gd', 'load':[None, './transfer/mnist_conv_weights.npy']}
mnist_conv_dfa = {'benchmark':'mnist_conv.py', 'epochs':300, 'batch_size':64, 'alpha':[0.01, 0.005], 'dfa':1, 'sparse':[0, 1], 'rank':0, 'init':'zero', 'opt':'gd', 'load':[None, './transfer/mnist_conv_weights.npy']}

cifar10_conv_bp = {'benchmark':'cifar10_conv.py', 'epochs':300, 'batch_size':64, 'alpha':[0.0001, 0.00003, 0.00001], 'dfa':0, 'sparse':0, 'rank':0, 'init':'sqrt_fan_in', 'opt':['adam'], 'load':[None], 'eps':[1e-4, 1e-6, 1e-8]}
cifar10_conv_dfa = {'benchmark':'cifar10_conv.py', 'epochs':300, 'batch_size':64, 'alpha':[0.0001, 0.00003, 0.00001], 'dfa':1, 'sparse':[0, 1], 'rank':0, 'init':'zero', 'opt':['adam'], 'load':[None], 'eps':[1e-4, 1e-6, 1e-8]}

cifar100_conv_bp = {'benchmark':'cifar100_conv.py', 'epochs':300, 'batch_size':64, 'alpha':[0.01, 0.005], 'dfa':0, 'sparse':0, 'rank':0, 'init':'sqrt_fan_in', 'opt':['gd', 'adam'], 'load':[None]}
cifar100_conv_dfa = {'benchmark':'cifar100_conv.py', 'epochs':300, 'batch_size':64, 'alpha':[0.01, 0.005], 'dfa':1, 'sparse':[0, 1], 'rank':0, 'init':'zero', 'opt':['gd', 'adam'], 'load':[None]}
'''

################################################
'''
imagenet_alexnet_bp = {'benchmark':'imagenet.py', 'epochs':100, 'batch_size':128, 'alpha':[0.01], 'dfa':0, 'sparse':0, 'rank':0, 'init':['NA', 'alexnet'], 'opt':'gd', 'load':None}
imagenet_vgg_bp = {'benchmark':'imagenet_vgg.py', 'epochs':100, 'batch_size':32, 'alpha':[0.01], 'dfa':0, 'sparse':0, 'rank':0, 'init':['sqrt_fan_in'], 'opt':'gd', 'load':None}
'''
################################################

'''
params = [mnist_fc_bp,      \
          mnist_fc_dfa,     \
          cifar10_fc_bp,    \
          cifar10_fc_dfa,   \
          cifar100_fc_bp,   \
          cifar100_fc_dfa,  \
          mnist_conv_bp,    \
          mnist_conv_dfa,   \
          cifar10_conv_bp,  \ 
          cifar10_conv_dfa, \
          cifar100_conv_bp, \
          cifar100_conv_dfa]
'''

# params = [mnist_fc_dfa, cifar10_conv_dfa]
# params = [imagenet_alexnet_bp, imagenet_vgg_bp]
# params = [imagenet_alexnet_bp]
# params = [imagenet_vgg_bp]

params = [cifar10_fc_bp, cifar10_fc_bp1, cifar10_fc_dfa, cifar10_fc_dfa1]
# params = [cifar10_fc_dfa]
# params = [cifar10_fc_bp]

################################################

def get_runs():
    runs = []

    for param in params:
        perms = get_perms(param)
        runs.extend(perms)

    return runs

################################################

        
