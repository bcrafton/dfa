
import numpy as np
import os
import copy
import threading
import argparse

from results import get_runs

##############################################

num_gpus = 3
counter = 1

def run_command(param):
    global num_gpus, counter

    if num_gpus == 0:
        gpu = -1
    else:
        gpu = counter % num_gpus
        counter = counter + 1
    
    name = '%s_%f_%f_%s_%f_%f_%d_%d_%f_%s_%s' % (param['benchmark'], param['alpha'], param['eps'], param['act'], param['bias'], param['dropout'], param['dfa'], param['sparse'], param['rate'], param['init'], param['opt'])
    if param['load']:
        name += '_transfer'
        cmd = "python %s --gpu %d --epochs %d --batch_size %d --alpha %f --eps %f --act %s --bias %f --dropout %f --dfa %d --sparse %d --rank %d --rate %f --init %s --opt %s --save %d --name %s --load %s" % \
              (param['benchmark'], gpu, param['epochs'], param['batch_size'], param['alpha'], param['eps'], param['act'], param['bias'], param['dropout'], param['dfa'], param['sparse'], param['rank'], param['rate'], param['init'], param['opt'], 1, name, param['load'])
    else:
        cmd = "python %s --gpu %d --epochs %d --batch_size %d --alpha %f --eps %f --act %s --bias %f --dropout %f --dfa %d --sparse %d --rank %d --rate %f --init %s --opt %s --save %d --name %s" % \
              (param['benchmark'], gpu, param['epochs'], param['batch_size'], param['alpha'], param['eps'], param['act'], param['bias'], param['dropout'], param['dfa'], param['sparse'], param['rank'], param['rate'], param['init'], param['opt'], 1, name)

    os.system(cmd)

    return

##############################################

runs = get_runs()

##############################################

num_runs = len(runs)
parallel_runs = num_gpus

for run in range(0, num_runs, parallel_runs):
    threads = []
    for parallel_run in range( min(parallel_runs, num_runs - run)):
        args = runs[run + parallel_run]
        t = threading.Thread(target=run_command, args=(args,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
        
