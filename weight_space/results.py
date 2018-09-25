
import numpy as np
import os
import threading

def run_command():
    cmd = "python mnist_bp_batch.py --epochs 25 --batch_size 32 --alpha 0.01 --shuffle 1 --num " + str(run + parallel_run) + " --load 1"
    os.system(cmd)
    return

################################################

num_runs = 8
parallel_runs = 2

for run in range(0, num_runs, parallel_runs):
    threads = []
    for parallel_run in range(parallel_runs):
        t = threading.Thread(target=run_command)
        threads.append(t)
        t.start()
        
    for t in threads:
        t.join()
