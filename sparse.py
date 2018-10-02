
import numpy as np
import os
import threading

def run_command(sparse, rank, name, num):
    # use --gpu 1 so it dosnt see any gpu
    cmd = "python mnist_fc.py --epochs 100 --batch_size 32 --alpha 0.01 --gpu 1 --dfa 1 --sparse %d --rank %d --init zero --opt adam --save 1 --name %s --num %d --shuffle 1" % (sparse, rank, name, num)
    os.system(cmd)
    return

################################################

sparses = [1]
ranks = [10]
itrs = range(1, 100 + 1, 1)

runs = []
for sparse in sparses:
    for rank in ranks:
        for itr in itrs:
            runs.append((sparse, rank, "sparse%drank%ditr%d" % (sparse, rank, itr), itr))

num_runs = len(runs)
parallel_runs = 4

for run in range(0, num_runs, parallel_runs):
    threads = []
    for parallel_run in range(parallel_runs):
        args = runs[run + parallel_run]
        t = threading.Thread(target=run_command, args=args)
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
