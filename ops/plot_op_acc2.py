
import numpy as np
import math
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
import argparse

#######################################


sparsity = [1, 5, 10, 25, 50, 100, 1000, 'BP']
accuracy = np.array([0.63, 0.64, 0.65, 0.655, 0.66, 0.665, 0.67, 0.68]) * 100.
MACs = [58630144, 58662912, 58703872, 58826752, 59031552, 59441152, 66813952, 117243904]
movement = [9192, 45960, 91920, 229800, 459600, 919200, 9192000, 58632144]

'''
Sparsity	Accuracy	MACs	Data Movement
1	0.63	58630144	9192
5	0.64	58662912	45960
10	0.65	58703872	91920
25	0.655	58826752	229800
50	0.66	59031552	459600
100	0.665	59441152	919200
1000	0.67	66813952	9192000
BP	0.68	58631144	58632144
'''

#######################################

f, ax = plt.subplots(1, 1, sharex=False, sharey=False)

for idx in range(len(sparsity)):
    label = 'Sparse-' + str(sparsity[idx])
    ax.semilogy(accuracy[idx], movement[idx], '.', label=label, marker='.', color='#FFA500')
    # ax.annotate(' ' + str(sparsity[idx]), (accuracy[idx], movement[idx]))

for idx in range(len(sparsity)):
    label = 'Sparse-' + str(sparsity[idx])
    ax.semilogy(accuracy[idx], MACs[idx], '.', label=label, marker='.', color='#2020FF')
    # ax.annotate(' ' + str(sparsity[idx]), (accuracy[idx], MACs[idx]))

#######################################

f.subplots_adjust(hspace=0.0)
f.set_size_inches(3.5, 3.5)

plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['font.size'] = 10.

ax.set_xticks(np.linspace(62, 69, 8))
ax.set_yticks([1e4, 1e5, 1e6, 1e7, 1e8])

plt.xlabel('Accuracy (%)', fontsize=10)
plt.xticks(fontsize=10)

plt.ylabel('Operations (Words)', fontsize=10)
plt.yticks(fontsize=10)

plt.savefig('ops_vs_acc.png', bbox_inches='tight', dpi=1000)










