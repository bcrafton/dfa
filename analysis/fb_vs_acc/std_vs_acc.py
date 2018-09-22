
import numpy as np
import math
from scipy import stats
import matplotlib.pyplot as plt

#######################################
    
sparse_dfa_stds = []
sparse_dfa_acc = []

for ii in range(250):
    print (ii)
    
    
    acc = np.load("../results/random_feedback/sparse_dfa/acc_" + str(ii+1) + "_2.npy")
    sparse_dfa_acc.append(np.max(acc))
     
    B = np.load("../results/random_feedback/sparse_dfa/B_" + str(ii+1) + "_2.npy")
    stds = np.zeros(10)
    
    for ii in range(10):
        stds[ii] = np.std(B[ii])
            
    sparse_dfa_stds.append(np.std(stds))

#######################################

fit = np.poly1d(np.polyfit(sparse_dfa_acc, sparse_dfa_stds, 1))
pred = fit(sparse_dfa_acc)
plt.plot(sparse_dfa_acc, pred)

plt.plot(sparse_dfa_acc, sparse_dfa_stds, '.')
plt.show()
