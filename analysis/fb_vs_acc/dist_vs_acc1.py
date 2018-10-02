
import numpy as np
import math
from scipy import stats
import matplotlib.pyplot as plt

#######################################
    
sparse_dfa_dist = []
sparse_dfa_acc = []

for ii in range(99):
    print (ii)
    
    w = np.load("../../sparse/sparse1rank10itr" + str(ii+1) + ".npy").item()
    B = np.transpose(w['fc1_fb'])
    acc = np.max(w['acc'])
    
    dist = [0] * 10
    
    for ii in range(100):
        idx = np.argmax(B[ii] != 0.0)
        if dist[idx] < 10:
            dist[idx] += 1
            
    count = np.sum(dist)
    sparse_dfa_dist.append(count)
    
    sparse_dfa_acc.append(acc)

#######################################

fit = np.poly1d(np.polyfit(sparse_dfa_dist, sparse_dfa_acc, 1))
pred = fit(sparse_dfa_dist)
plt.plot(sparse_dfa_dist, pred)

plt.plot(sparse_dfa_dist, sparse_dfa_acc, '.')
plt.show()
