
import numpy as np
import math
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.cm as cmap

#######################################

'''
Lime 	              #00FF00 	(0,255,0)
Blue 	              #0000FF 	(0,0,255)
Yellow 	            #FFFF00 	(255,255,0)
Cyan / Aqua 	      #00FFFF 	(0,255,255)

Magenta / Fuchsia 	#FF00FF 	(255,0,255)
Silver 	            #C0C0C0 	(192,192,192)
Gray 	              #808080 	(128,128,128)
Maroon 	            #800000 	(128,0,0)

Teal 	              #008080 	(0,128,128)
orange 	            #FFA500 	(255,165,0)

Green 	            #008000 	(0,128,0)
Purple 	            #800080 	(128,0,128)
Navy 	              #000080 	(0,0,128)

Olive 	            #808000 	(128,128,0)
'''

colors = {1:  '#00FF00', \
          2:  '#2020FF', \
          3:  '#FFFF00', \
          4:  '#00FFFF', \
          
          5:  '#FF00FF', \
          6:  '#C0C0C0', \
          7:  '#808080', \
          8:  '#800000', \
          
          9:  '#008080', \
          10: '#FFA500'}

#######################################

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

#######################################

mnist = np.load('mnist_data.npy')
cifar10 = np.load('cifar10_data.npy')

#######################################

data_grouped = {}
ii = 0
for ii in range(len(mnist)):
    d = mnist[ii]
    key = d['sparse']
    
    if key in data_grouped.keys():
        data_grouped[key].append([d['rank'], d['acc']])
    else:
        data_grouped[key] = [[d['rank'], d['acc']]]
        
points1 = []
labels1 = [] 
for key in data_grouped:
    points1.append( data_grouped[key] )
    labels1.append( key )

#######################################

data_grouped = {}
ii = 0
for ii in range(len(mnist)):
    d = mnist[ii]
    key = d['sparse']
    
    if key in data_grouped.keys():
        data_grouped[key].append([d['rank'], d['angle']])
    else:
        data_grouped[key] = [[d['rank'], d['angle']]]
        
points2 = []
labels2 = [] 
for key in data_grouped:
    points2.append( data_grouped[key] )
    labels2.append( key )

#######################################

data_grouped = {}
ii = 0
for ii in range(len(cifar10)):
    d = cifar10[ii]
    key = d['sparse']
    
    if key in data_grouped.keys():
        data_grouped[key].append([d['rank'], d['acc']])
    else:
        data_grouped[key] = [[d['rank'], d['acc']]]
        
points3 = []
labels3 = [] 
for key in data_grouped:
    points3.append( data_grouped[key] )
    labels3.append( key )

#######################################

data_grouped = {}
ii = 0
for ii in range(len(cifar10)):
    d = cifar10[ii]
    key = d['sparse']
    
    if key in data_grouped.keys():
        data_grouped[key].append([d['rank'], d['angle']])
    else:
        data_grouped[key] = [[d['rank'], d['angle']]]
        
points4 = []
labels4 = [] 
for key in data_grouped:
    points4.append( data_grouped[key] )
    labels4.append( key )
#######################################

plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['font.size'] = 10

f, [[ax1, ax3], [ax2, ax4]] = plt.subplots(2, 2, sharex=True, sharey=False)

for ii in range(len(points1)):
    p = points1[ii]
    p = np.transpose(p)
    label = "%s %d" % ('Sparse', labels1[ii])
    ax1.scatter(p[0], p[1], s=10, label=label, color=colors[labels1[ii]])

for ii in range(len(points2)):
    p = points2[ii]
    p = np.transpose(p)
    label = "%s %d" % ('Sparse', labels2[ii])
    ax2.scatter(p[0], p[1], s=10, label=label, color=colors[labels1[ii]])

for ii in range(len(points3)):
    p = points3[ii]
    p = np.transpose(p)
    label = "%s %d" % ('Sparse', labels3[ii])
    ax3.scatter(p[0], p[1], s=10, label=label, color=colors[labels1[ii]])

for ii in range(len(points4)):
    p = points4[ii]
    p = np.transpose(p)
    label = "%s %d" % ('Sparse', labels4[ii])
    ax4.scatter(p[0], p[1], s=10, label=label, color=colors[labels1[ii]])

# ax2.set_xticks(range(1, 11))
# ax4.set_xticks(range(1, 11))

# ax1.set_yticks(np.linspace(.8, .98, 7))

plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['font.size'] = 10.

ax2.set_xticks(range(1, 10+1, 1))
ax4.set_xticks(range(1, 10+1, 1))

ax2.set_xlabel(xlabel='Rank', fontsize=10)
ax4.set_xlabel(xlabel='Rank', fontsize=10)

ax1.set_ylabel(ylabel='Accuracy', fontsize=10)
ax2.set_ylabel(ylabel='Angle', fontsize=10)
# ax3.set_ylabel(ylabel='Accuracy')
# ax4.set_ylabel(ylabel='Angle')

f.subplots_adjust(hspace=0)
f.set_size_inches(7., 5.)
# f.set_size_inches(10, 10)

# lgd = ax4.legend(loc='upper left', bbox_to_anchor=(1.02, 1.5), fontsize=10)

for ax in [ax1, ax2, ax3, ax4]:
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(10) 
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(10) 

f.savefig('plot1.svg', bbox_inches='tight', dpi=300)
# plt.show()




