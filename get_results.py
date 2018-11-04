
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

mnist_fc_bp = {'benchmark':'mnist_fc.py', 'epochs':300, 'batch_size':32, 'alpha':[0.05, 0.03], 'dfa':0, 'sparse':0, 'rank':0, 'init':'sqrt_fan_in', 'opt':'gd', 'load':None}
mnist_fc_dfa = {'benchmark':'mnist_fc.py', 'epochs':300, 'batch_size':32, 'alpha':[0.05, 0.03], 'dfa':1, 'sparse':[0, 1], 'rank':0, 'init':'zero', 'opt':'gd', 'load':None}

cifar10_fc_bp = {'benchmark':'cifar10_fc.py', 'epochs':300, 'batch_size':64, 'alpha':[0.01, 0.005], 'dfa':0, 'sparse':0, 'rank':0, 'init':'sqrt_fan_in', 'opt':'gd', 'load':None}
cifar10_fc_dfa = {'benchmark':'cifar10_fc.py', 'epochs':300, 'batch_size':64, 'alpha':[0.01, 0.005], 'dfa':1, 'sparse':[0, 1], 'rank':0, 'init':'zero', 'opt':'gd', 'load':None}

cifar100_fc_bp = {'benchmark':'cifar100_fc.py', 'epochs':300, 'batch_size':64, 'alpha':[0.01, 0.005], 'dfa':0, 'sparse':0, 'rank':0, 'init':'sqrt_fan_in', 'opt':'gd', 'load':None}
cifar100_fc_dfa = {'benchmark':'cifar100_fc.py', 'epochs':300, 'batch_size':64, 'alpha':[0.01, 0.005], 'dfa':1, 'sparse':[0, 1], 'rank':0, 'init':'zero', 'opt':'gd', 'load':None}

################################################

mnist_conv_bp = {'benchmark':'mnist_conv.py', 'epochs':300, 'batch_size':64, 'alpha':[0.01, 0.005], 'dfa':0, 'sparse':0, 'rank':0, 'init':'sqrt_fan_in', 'opt':'gd', 'load':[None, './transfer/mnist_conv_weights.npy']}
mnist_conv_dfa = {'benchmark':'mnist_conv.py', 'epochs':300, 'batch_size':64, 'alpha':[0.01, 0.005], 'dfa':1, 'sparse':[0, 1], 'rank':0, 'init':'zero', 'opt':'gd', 'load':[None, './transfer/mnist_conv_weights.npy']}

cifar10_conv_bp = {'benchmark':'cifar10_conv.py', 'epochs':300, 'batch_size':64, 'alpha':[0.005, 0.003], 'dfa':0, 'sparse':0, 'rank':0, 'init':'sqrt_fan_in', 'opt':'gd', 'load':[None, './transfer/cifar10_conv_weights.npy']}
cifar10_conv_dfa = {'benchmark':'cifar10_conv.py', 'epochs':300, 'batch_size':64, 'alpha':[0.005, 0.003], 'dfa':1, 'sparse':[0, 1], 'rank':0, 'init':'zero', 'opt':'gd', 'load':[None, './transfer/cifar10_conv_weights.npy']}

cifar100_conv_bp = {'benchmark':'cifar100_conv.py', 'epochs':300, 'batch_size':64, 'alpha':[0.01, 0.005], 'dfa':0, 'sparse':0, 'rank':0, 'init':'sqrt_fan_in', 'opt':['gd', 'adam'], 'load':[None, './transfer/cifar100_conv_weights.npy']}
cifar100_conv_dfa = {'benchmark':'cifar100_conv.py', 'epochs':300, 'batch_size':64, 'alpha':[0.01, 0.005], 'dfa':1, 'sparse':[0, 1], 'rank':0, 'init':'zero', 'opt':['gd', 'adam'], 'load':[None, './transfer/cifar100_conv_weights.npy']}

################################################

params = [mnist_fc_bp, mnist_fc_dfa, cifar10_fc_bp, cifar10_fc_dfa, cifar100_fc_bp, cifar100_fc_dfa, mnist_conv_bp, mnist_conv_dfa, cifar10_conv_bp, cifar10_conv_dfa, cifar100_conv_bp, cifar100_conv_dfa]

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
    # figure out the name of the param
    name = '%s_%f_%d_%d_%s_%s' % (param['benchmark'], param['alpha'], param['dfa'], param['sparse'], param['init'], param['opt'])
    if param['load']:
        name += '_transfer'
        cmd = "python %s --gpu %d --epochs %d --batch_size %d --alpha %f --dfa %d --sparse %d --rank %d --init %s --opt %s --save %d --name %s --load %s" % \
              (param['benchmark'], gpu, param['epochs'], param['batch_size'], param['alpha'], param['dfa'], param['sparse'], param['rank'], param['init'], param['opt'], 1, name, param['load'])
    else:
        cmd = "python %s --gpu %d --epochs %d --batch_size %d --alpha %f --dfa %d --sparse %d --rank %d --init %s --opt %s --save %d --name %s" % \
              (param['benchmark'], gpu, param['epochs'], param['batch_size'], param['alpha'], param['dfa'], param['sparse'], param['rank'], param['init'], param['opt'], 1, name)
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
        if results[key] < val:
            results[key] = val
    else:
        results[key] = val
            
for key in sorted(results.keys()):   
    print (key, results[key])





        
