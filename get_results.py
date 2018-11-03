
import numpy as np
import os
import copy
import threading
import argparse

##############################################

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

mnist_fc_bp = {'benchmark':'mnist_fc.py', 'epochs':300, 'batch_size':32, 'alpha':[0.01, 0.005], 'dfa':0, 'sparse':0, 'rank':0, 'init':'sqrt_fan_in', 'opt':'gd'}
mnist_fc_dfa = {'benchmark':'mnist_fc.py', 'epochs':300, 'batch_size':32, 'alpha':[0.01, 0.005], 'dfa':1, 'sparse':[0, 1], 'rank':0, 'init':'zero', 'opt':'gd'}

cifar10_fc_bp = {'benchmark':'cifar10_fc.py', 'epochs':300, 'batch_size':64, 'alpha':[0.01, 0.005], 'dfa':0, 'sparse':0, 'rank':0, 'init':'sqrt_fan_in', 'opt':'gd'}
cifar10_fc_dfa = {'benchmark':'cifar10_fc.py', 'epochs':300, 'batch_size':64, 'alpha':[0.01, 0.005], 'dfa':1, 'sparse':[0, 1], 'rank':0, 'init':'zero', 'opt':'gd'}

cifar100_fc_bp = {'benchmark':'cifar100_fc.py', 'epochs':300, 'batch_size':64, 'alpha':[0.01, 0.005], 'dfa':0, 'sparse':0, 'rank':0, 'init':'sqrt_fan_in', 'opt':'gd'}
cifar100_fc_dfa = {'benchmark':'cifar100_fc.py', 'epochs':300, 'batch_size':64, 'alpha':[0.01, 0.005], 'dfa':1, 'sparse':[0, 1], 'rank':0, 'init':'zero', 'opt':'gd'}

################################################

mnist_conv_bp = {'benchmark':'mnist_conv.py', 'epochs':300, 'batch_size':64, 'alpha':[0.01, 0.005], 'dfa':0, 'sparse':0, 'rank':0, 'init':'sqrt_fan_in', 'opt':'gd'}
mnist_conv_dfa = {'benchmark':'mnist_conv.py', 'epochs':300, 'batch_size':64, 'alpha':[0.01, 0.005], 'dfa':1, 'sparse':[0, 1], 'rank':0, 'init':'zero', 'opt':'gd'}

cifar10_conv_bp = {'benchmark':'cifar10_conv.py', 'epochs':300, 'batch_size':64, 'alpha':[0.01, 0.005], 'dfa':0, 'sparse':0, 'rank':0, 'init':'sqrt_fan_in', 'opt':'gd'}
cifar10_conv_dfa = {'benchmark':'cifar10_conv.py', 'epochs':300, 'batch_size':64, 'alpha':[0.01, 0.005], 'dfa':1, 'sparse':[0, 1], 'rank':0, 'init':'zero', 'opt':'gd'}

cifar100_conv_bp = {'benchmark':'cifar100_conv.py', 'epochs':300, 'batch_size':64, 'alpha':[0.01, 0.005], 'dfa':0, 'sparse':0, 'rank':0, 'init':'sqrt_fan_in', 'opt':'gd'}
cifar100_conv_dfa = {'benchmark':'cifar100_conv.py', 'epochs':300, 'batch_size':64, 'alpha':[0.01, 0.005], 'dfa':1, 'sparse':[0, 1], 'rank':0, 'init':'zero', 'opt':'gd'}

################################################

# params = [mnist_fc_bp, mnist_fc_dfa, cifar10_fc_bp, cifar10_fc_dfa, cifar100_fc_bp, cifar100_fc_dfa, mnist_conv_bp, mnist_conv_dfa, cifar10_conv_bp, cifar10_conv_dfa, cifar100_conv_bp, cifar100_conv_dfa]
params = [mnist_fc_bp, mnist_fc_dfa, cifar10_fc_bp, cifar10_fc_dfa, mnist_conv_bp, mnist_conv_dfa, cifar10_conv_bp, cifar10_conv_dfa, cifar100_conv_bp, cifar100_conv_dfa]

################################################

runs = []

for param in params:
    perms = get_perms(param)
    runs.extend(perms)
    
################################################

results = {}

num_runs = len(runs)
for ii in range(num_runs):
    param = runs[ii]
    name = '%s_%f_%d_%d_%s_%s.npy' % (param['benchmark'], param['alpha'], param['dfa'], param['sparse'], param['init'], param['opt'])
    res = np.load(name)
    
    key = (param['benchmark'], param['dfa'], param['sparse'])
    val = res['test_acc']
    
    if key in results.keys():
        # use an if instead of max because we gonna want to save the winner run information
        if results[key] < val:
            results[key] = val
    else:
        results[key] = val
            
    






        
