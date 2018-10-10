
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--benchmark', type=str, default='mnist')
parser.add_argument('--itrs', type=int, default=10)
args = parser.parse_args()

benchmark = args.benchmark
itrs = range(1, args.itrs+1)

fig = plt.figure(figsize=(10, 10))

for sparse in range(1, 10+1):
    ranks = []
    accs = []
    
    for rank in range(sparse, 10+1):
        for itr in itrs:
            fname = benchmark + "/sparse%drank%ditr%d.npy" % (sparse, rank, itr)
            results = np.load(fname).item()
            acc = np.max(results['acc'])

            print ("sparse %d rank %d itr %d acc %f" % (sparse, rank, itr, acc))

            accs.append(acc)
            ranks.append(rank)
       
    scatter = plt.scatter(ranks, accs, s=10, label="Sparse " + str(sparse))
     
plt.xlabel("Rank", fontsize=18)
plt.xticks(fontsize=14)

plt.ylabel("Accuracy", fontsize=18)
plt.yticks(fontsize=14)

plt.legend(fontsize=18, markerscale=4.0)
plt.savefig(benchmark + '_sparse_vs_rank.png')
