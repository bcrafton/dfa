
import numpy as np
import os
import threading

num_gpus = 4
counter = 0

def run_command(benchmark, sparse, rank, name):
    global num_gpus, counter

    if num_gpus == 0:
        gpu = -1
    else:
        gpu = counter % num_gpus
        counter = counter + 1
    
    cmd = "python %s --epochs 200 --batch_size 64 --alpha 0.0025 --gpu %d --dfa 1 --sparse %d --rank %d --init zero --opt adam --save 1 --name %s" % (benchmark, gpu, sparse, rank, name)
    os.system(cmd)
    # print (cmd)
    return

################################################

benchmark = 'mnist_fc.py'
benchmark = 'cifar10_fc.py'

runs = []
for sparse in range(1, 10+1, 1):
    for rank in range(sparse, 10+1, 1):
        for itr in range(1, 10+1, 1):
            runs.append((benchmark, sparse, rank, "sparse%drank%ditr%d" % (sparse, rank, itr)))

num_runs = len(runs)
parallel_runs = 8

for run in range(0, num_runs, parallel_runs):
    threads = []
    for parallel_run in range( min(parallel_runs, num_runs - run)):
        args = runs[run + parallel_run]
        t = threading.Thread(target=run_command, args=args)
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
