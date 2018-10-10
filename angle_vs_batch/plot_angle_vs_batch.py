
import numpy as np
import matplotlib.pyplot as plt

sparses = [0, 1]
ranks = [0]
itrs = [0]

accs = []
angles = []

fig = plt.figure(figsize=(10, 10))

for sparse in sparses:    
    for rank in ranks:
        for itr in itrs:
            fname = "./angle_vs_batch/angles_sparse%drank%ditr%d.npy" % (sparse, rank, itr)
            results = np.load(fname).item()
            
            accs = results['acc']
            angles = results['angles'] 

            epochs = results['epochs']
            batches = results['batches']

            print ("sparse %d rank %d itr %d" % (sparse, rank, itr))

            if sparse:
                label = "Sparse"
            else:
                label = "Full"
                
            # scatter = plt.scatter(epochs, accs, s=10, label=label)
            scatter = plt.scatter(batches, angles, s=10, label=label)
     
     
plt.xlabel("Batch", fontsize=18)
plt.xticks(fontsize=14)

plt.ylabel("Angle", fontsize=18)
plt.yticks(fontsize=14)
# plt.ylim([.9, 1])

plt.legend(fontsize=18, markerscale=4.0)
plt.savefig('angle_vs_batch')
# plt.savefig('acc_vs_epoch')
