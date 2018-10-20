
import numpy as np
import os
import threading

num_gpus = 4
counter = 0

def run_command(sparse, name):
    global num_gpus, counter

    if num_gpus == 0:
        gpu = -1
    else:
        gpu = counter % num_gpus
        counter = counter + 1
    
    cmd = "python mnist_fc.py --epochs 100 --batch_size 32 --alpha 0.01 --gpu %d --dfa 1 --sparse %d --rank 0 --init zero --opt gd --save 1 --name %s" % (gpu, sparse, name)
    os.system(cmd)
    return

################################################

sparses = [0, 1]
itrs = range(500)

runs = []
for sparse in sparses:
        for itr in itrs:
            runs.append((sparse, "sparse%ditr%d" % (sparse, itr)))

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
