
import numpy as np
import math
from scipy import stats
import matplotlib.pyplot as plt

#######################################
    
sparse_dfa_dist = []
sparse_dfa_acc = []

for ii in range(250):
    print (ii)
    
    
    acc = np.load("../results/random_feedback/sparse_dfa/acc_" + str(ii+1) + "_2.npy")
    sparse_dfa_acc.append(np.max(acc))
     
    B = np.load("../results/random_feedback/sparse_dfa/B_" + str(ii+1) + "_2.npy")
    B = np.transpose(B)
    
    dist = [0] * 10
    
    for ii in range(100):
        idx = np.argmax(B[ii] != 0.0)
        if dist[idx] < 10:
            dist[idx] += 1
            
    count = np.sum(dist)
    sparse_dfa_dist.append(count)

#######################################

fit = np.poly1d(np.polyfit(sparse_dfa_acc, sparse_dfa_dist, 1))
pred = fit(sparse_dfa_acc)
plt.plot(sparse_dfa_acc, pred)

plt.plot(sparse_dfa_acc, sparse_dfa_dist, '.')
plt.show()
