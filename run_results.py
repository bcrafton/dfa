
import numpy as np
import os
import copy
import threading
import argparse

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--make_filters', type=int, default=1)
args = parser.parse_args()

##############################################

num_gpus = 4
counter = 0

def run_command(param):
    global num_gpus, counter

    if num_gpus == 0:
        gpu = -1
    else:
        gpu = counter % num_gpus
        counter = counter + 1
    
    name = '%s_%f_%d_%d_%s_%s' % (param['benchmark'], param['alpha'], param['dfa'], param['sparse'], param['init'], param['opt'])
    
    cmd = "python %s --gpu %d --epochs %d --batch_size %d --alpha %f --dfa %d --sparse %d --rank %d --init %s --opt %s --save %d --name %s --load %s" % \
          (param['benchmark'], gpu, param['epochs'], param['batch_size'], param['alpha'], param['dfa'], param['sparse'], param['rank'], param['init'], param['opt'], 1, name, param['load'])

    # print cmd
    os.system(cmd)

    return

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

mnist_fc_bp = {'benchmark':'mnist_fc.py', 'epochs':300, 'batch_size':32, 'alpha':[0.01, 0.005], 'dfa':0, 'sparse':0, 'rank':0, 'init':'sqrt_fan_in', 'opt':'gd', 'load':''}
mnist_fc_dfa = {'benchmark':'mnist_fc.py', 'epochs':300, 'batch_size':32, 'alpha':[0.01, 0.005], 'dfa':1, 'sparse':[0, 1], 'rank':0, 'init':'zero', 'opt':'gd', 'load':''}

cifar10_fc_bp = {'benchmark':'cifar10_fc.py', 'epochs':300, 'batch_size':64, 'alpha':[0.01, 0.005], 'dfa':0, 'sparse':0, 'rank':0, 'init':'sqrt_fan_in', 'opt':'gd', 'load':''}
cifar10_fc_dfa = {'benchmark':'cifar10_fc.py', 'epochs':300, 'batch_size':64, 'alpha':[0.01, 0.005], 'dfa':1, 'sparse':[0, 1], 'rank':0, 'init':'zero', 'opt':'gd', 'load':''}

cifar100_fc_bp = {'benchmark':'cifar100_fc.py', 'epochs':300, 'batch_size':64, 'alpha':[0.01, 0.005], 'dfa':0, 'sparse':0, 'rank':0, 'init':'sqrt_fan_in', 'opt':'gd', 'load':''}
cifar100_fc_dfa = {'benchmark':'cifar100_fc.py', 'epochs':300, 'batch_size':64, 'alpha':[0.01, 0.005], 'dfa':1, 'sparse':[0, 1], 'rank':0, 'init':'zero', 'opt':'gd', 'load':''}

################################################

mnist_conv_bp = {'benchmark':'mnist_conv.py', 'epochs':300, 'batch_size':64, 'alpha':[0.01, 0.005], 'dfa':0, 'sparse':0, 'rank':0, 'init':'sqrt_fan_in', 'opt':'gd', 'load':['', './transfer/mnist_conv_weights.npy']}
mnist_conv_dfa = {'benchmark':'mnist_conv.py', 'epochs':300, 'batch_size':64, 'alpha':[0.01, 0.005], 'dfa':1, 'sparse':[0, 1], 'rank':0, 'init':'zero', 'opt':'gd', 'load':['', './transfer/mnist_conv_weights.npy']}

cifar10_conv_bp = {'benchmark':'cifar10_conv.py', 'epochs':300, 'batch_size':64, 'alpha':[0.01, 0.005], 'dfa':0, 'sparse':0, 'rank':0, 'init':'sqrt_fan_in', 'opt':'gd', 'load':['', './transfer/cifar10_conv_weights.npy']}
cifar10_conv_dfa = {'benchmark':'cifar10_conv.py', 'epochs':300, 'batch_size':64, 'alpha':[0.01, 0.005], 'dfa':1, 'sparse':[0, 1], 'rank':0, 'init':'zero', 'opt':'gd', 'load':['', './transfer/cifar10_conv_weights.npy']}

cifar100_conv_bp = {'benchmark':'cifar100_conv.py', 'epochs':300, 'batch_size':64, 'alpha':[0.01, 0.005], 'dfa':0, 'sparse':0, 'rank':0, 'init':'sqrt_fan_in', 'opt':['gd', 'adam'], 'load':['', './transfer/cifar100_conv_weights.npy']}
cifar100_conv_dfa = {'benchmark':'cifar100_conv.py', 'epochs':300, 'batch_size':64, 'alpha':[0.01, 0.005], 'dfa':1, 'sparse':[0, 1], 'rank':0, 'init':'zero', 'opt':['gd', 'adam'], 'load':['', './transfer/cifar100_conv_weights.npy']}

################################################

if args.make_filters:
    params = [mnist_conv_bp, cifar10_conv_bp, cifar100_conv_bp]
else:
    params = [mnist_fc_bp, mnist_fc_dfa, cifar10_fc_bp, cifar10_fc_dfa, cifar100_fc_bp, cifar100_fc_dfa, \
              mnist_conv_bp, mnist_conv_dfa, cifar10_conv_bp, cifar10_conv_dfa, cifar100_conv_bp, cifar100_conv_dfa]

################################################

runs = []

for param in params:
    perms = get_perms(param)
    runs.extend(perms)
    
################################################

num_runs = len(runs)
parallel_runs = 8

for run in range(0, num_runs, parallel_runs):
    threads = []
    for parallel_run in range( min(parallel_runs, num_runs - run)):
        args = runs[run + parallel_run]
        t = threading.Thread(target=run_command, args=(args,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
        
