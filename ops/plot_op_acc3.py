
import numpy as np
import math
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
import argparse

#######################################


'''
sparsity = [1, 5, 10, 25, 50, 100, 1000, 'BP']
accuracy = np.array([0.63, 0.64, 0.65, 0.655, 0.66, 0.665, 0.67, 0.68]) * 100.
MACs = [58630144, 58662912, 58703872, 58826752, 59031552, 59441152, 66813952, 58631144]
movement = [9192, 45960, 91920, 229800, 459600, 919200, 9192000, 58632144]

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
'''
#######################################

f, [ax1, ax2] = plt.subplots(2, 1, sharex=True, sharey=False)

ax1.bar(accuracy, movement, width=0.5)

ax2.bar(accuracy, MACs, width=0.5)

# for idx in range(len(sparsity)):
    # ax1.semilogy(accuracy[idx], movement[idx], '.', marker='.', color='#FFA500')

# for idx in range(len(sparsity)):
    # ax2.semilogy(accuracy[idx], MACs[idx], '.', marker='.', color='#2020FF')

    
#######################################

f.subplots_adjust(hspace=0.0)
f.set_size_inches(3.5, 5)

plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['font.size'] = 10.

# ax1.set_xticks(np.linspace(62, 69, 8))
# ax1.set_yticks([1e4, 1e5, 1e6, 1e7, 1e8])

# ax2.set_xticks(np.linspace(62, 69, 8))
# ax2.set_yticks([1e4, 1e5, 1e6, 1e7, 1e8])

plt.xlabel('Accuracy (%)', fontsize=10)
plt.xticks(fontsize=10)

plt.ylabel('Data Movement (Words)', fontsize=10)
plt.yticks(fontsize=10)

plt.savefig('ops_vs_acc.png', bbox_inches='tight', dpi=300)
'''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple

sparsity = [1, 5, 10, 25, 50, 100, 1000, 'BP']
accuracy = np.array([0.63, 0.64, 0.65, 0.655, 0.66, 0.665, 0.67, 0.68]) * 100.
MACs = [58630144, 58662912, 58703872, 58826752, 59031552, 59441152, 66813952, 117243904]
movement = [9192, 45960, 91920, 229800, 459600, 919200, 9192000, 58632144]

n_groups = len(sparsity)

fig, ax = plt.subplots()

index = np.arange(n_groups)
bar_width = 0.35

rects1 = ax.bar(index, MACs, bar_width, color='b')

ax.set_xlabel('Group')
ax.set_ylabel('Scores')
ax.set_title('Scores by group and gender')
ax.set_xticks(index)
# ax.set_yticks([1e7, 1e8])
ax.set_xticklabels(sparsity)
ax.legend()

fig.tight_layout()
plt.ticklabel_format(style='sci', axis='y', useOffset=False)
plt.show()








