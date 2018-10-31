
import numpy as np
import math
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.cm as cmap

#######################################

f, [[ax1, ax3], [ax2, ax4]] = plt.subplots(2, 2, sharex=True, sharey=False)

#######################################

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

#######################################

cifar10 = np.load('cifar10_data.npy')
mnist = np.load('mnist_data.npy')

#######################################

print ("mnist acc")

data = []

ii = 0
for ii in range(len(mnist)):
    d = mnist[ii]
    if d['rank'] == 10 or d['rank'] == 5:
        data.append(d)

data_grouped = {}
ii = 0
for ii in range(len(data)):
    d = data[ii]
    key = d['rank']
    
    if key in data_grouped.keys():
        data_grouped[key].append([d['sparse'], d['acc']])
    else:
        data_grouped[key] = [[d['sparse'], d['acc']]]
        
points = []
labels = [] 
for key in data_grouped:
    points.append( data_grouped[key] )
    labels.append( key )

for ii in range(len(points)):
    p = points[ii]
    p = np.transpose(p)
    label = "%s %d" % ('Sparse', labels[ii])
    
    ax1.scatter(p[0], p[1], s=10, label=label)

#######################################

print ("mnist angle")

data = []

ii = 0
for ii in range(len(mnist)):
    d = mnist[ii]
    if d['rank'] == 10 or d['rank'] == 5:
        data.append(d)

data_grouped = {}
ii = 0
for ii in range(len(data)):
    d = data[ii]
    key = d['rank']

    if key in data_grouped.keys():
        data_grouped[key].append([d['sparse'], d['angle']])
    else:
        data_grouped[key] = [[d['sparse'], d['angle']]]

points = []
labels = []
for key in data_grouped:
    points.append( data_grouped[key] )
    labels.append( key )

for ii in range(len(points)):
    p = points[ii]
    p = np.transpose(p)
    label = "%s %d" % ('Sparse', labels[ii])

    ax2.scatter(p[0], p[1], s=10, label=label)

#######################################

print ("cifar10 acc")

data = []

ii = 0
for ii in range(len(cifar10)):
    d = cifar10[ii]
    if d['rank'] == 10 or d['rank'] == 5:
        data.append(d)

data_grouped = {}
ii = 0
for ii in range(len(data)):
    d = data[ii]
    key = d['rank']
    
    if key in data_grouped.keys():
        data_grouped[key].append([d['sparse'], d['acc']])
    else:
        data_grouped[key] = [[d['sparse'], d['acc']]]
        
points = []
labels = [] 
for key in data_grouped:
    points.append( data_grouped[key] )
    labels.append( key )

for ii in range(len(points)):
    p = points[ii]
    p = np.transpose(p)
    label = "%s %d" % ('Sparse', labels[ii])
    
    ax3.scatter(p[0], p[1], s=10, label=label)

#######################################

print ("cifar10 angle")

data = []

ii = 0
for ii in range(len(cifar10)):
    d = cifar10[ii]
    if d['rank'] == 10 or d['rank'] == 5:
        data.append(d)

data_grouped = {}
ii = 0
for ii in range(len(data)):
    d = data[ii]
    key = d['rank']

    if key in data_grouped.keys():
        data_grouped[key].append([d['sparse'], d['angle']])
    else:
        data_grouped[key] = [[d['sparse'], d['angle']]]

points = []
labels = []
for key in data_grouped:
    points.append( data_grouped[key] )
    labels.append( key )

for ii in range(len(points)):
    p = points[ii]
    p = np.transpose(p)
    label = "%s %d" % ('Sparse', labels[ii])

    ax4.scatter(p[0], p[1], s=10, label=label)

#######################################

plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['font.size'] = 10

# ax1.set_yticks(np.linspace(.8, .98, 7))

ax2.set_xlabel(xlabel='Sparsity')
ax4.set_xlabel(xlabel='Sparsity')

ax1.set_ylabel(ylabel='Accuracy')
ax2.set_ylabel(ylabel='Angle')
# ax3.set_ylabel(ylabel='Accuracy')
# ax4.set_ylabel(ylabel='Angle')

f.set_size_inches(8., 6.)

for ax in [ax1, ax2, ax3, ax4]:
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(8) 
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(8) 

lgd = ax4.legend(loc='upper left', bbox_to_anchor=(1.02, 1.5), fontsize=8)

f.subplots_adjust(hspace=0)
f.savefig('plot2', bbox_extra_artists=(lgd,), bbox_inches='tight')



