
import numpy as np
import matplotlib.pyplot as plt

# for sparse in range(1, 10+1):
for sparse in range(1, 10+1):
    ranks = []
    accs = []
    
    for rank in range(sparse, 10+1):
        for itr in range(1, 10+1):
            fname = "./sparse_rank_results1/sparse%drank%ditr%d.npy" % (sparse, rank, itr)
            results = np.load(fname).item()
            acc = np.max(results['acc'])

            print ("sparse %d rank %d itr %d acc %f" % (sparse, rank, itr, acc))

            accs.append(acc)
            ranks.append(rank)
       
    scatter = plt.scatter(ranks, accs, s=10, label="Sparse " + str(sparse))
     
plt.xlabel("Rank")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
