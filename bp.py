
import numpy as np
import os
import threading

def run_command(name, num):
    cmd = "python mnist_fc.py --epochs 100 --batch_size 32 --alpha 0.01 --gpu -1 --dfa 0 --sparse 0 --rank 0 --init sqrt_fan_in --opt adam --save 1 --name %s --num %d --shuffle 1" % (name, num)
    os.system(cmd)
    return

################################################

sparses = [0]
ranks = [0]
itrs = range(1, 1001, 1)

runs = []
for sparse in sparses:
    for rank in ranks:
        for itr in itrs:
            runs.append(("itr%d" % (itr), itr))

num_runs = len(runs)
parallel_runs = 8

for run in range(0, num_runs, parallel_runs):
    threads = []
    for parallel_run in range(parallel_runs):
        args = runs[run + parallel_run]
        t = threading.Thread(target=run_command, args=args)
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
