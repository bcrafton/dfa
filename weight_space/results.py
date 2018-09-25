
import numpy as np
import os

num_runs = 8
parallel_runs = 4

for run in range(0, num_runs, parallel_runs):
    for parallel_run in range(parallel_runs):
        cmd = "python mnist_bp_batch.py --epochs 25 --batch_size 32 --alpha 0.01 --shuffle 1 --num " + str(run + parallel_run) + " &"
        os.system(cmd)

