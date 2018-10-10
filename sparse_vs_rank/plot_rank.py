
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--benchmark', type=str, default='mnist')
parser.add_argument('--itrs', type=int, default=10)
parser.add_argument('--rank', type=int, default=10)
args = parser.parse_args()

benchmark = args.benchmark
itrs = range(1, args.itrs+1)
rank = args.rank
sparses = []
accs = []

fig = plt.figure(figsize=(10, 10))

for sparse in range(1, rank + 1): 
    for itr in itrs:
        fname = benchmark + "/sparse%drank%ditr%d.npy" % (sparse, rank, itr)
        results = np.load(fname).item()
        acc = np.max(results['acc'])

        print ("sparse %d rank %d itr %d acc %f" % (sparse, rank, itr, acc))

        accs.append(acc)
        sparses.append(sparse)
       
    scatter = plt.scatter(sparses, accs, s=10, label="Sparse " + str(sparse))
     
plt.xlabel("Sparse", fontsize=18)
plt.xticks(fontsize=14)

plt.ylabel("Accuracy", fontsize=18)
plt.yticks(fontsize=14)

plt.legend(fontsize=18, markerscale=4.0)
plt.savefig(benchmark + '_plot_rank' + str(rank) + '.png')
