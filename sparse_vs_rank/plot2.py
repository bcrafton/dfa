
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

Olive 	            #808000 	(128,128,0)
Teal 	              #008080 	(0,128,128)

Green 	            #008000 	(0,128,0)
Purple 	            #800080 	(128,0,128)
Navy 	              #000080 	(0,0,128)

orange 	            #FFA500 	(255,165,0)
'''

colors = {5:  '#FFA500', \
          10: '#2020FF'}

#######################################

plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['font.size'] = 10.

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
    label = "%s %d" % ('Rank', labels[ii])
    
    ax1.scatter(p[0], p[1], s=10, label=label, color=colors[labels[ii]])

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
    label = "%s %d" % ('Rank', labels[ii])

    ax2.scatter(p[0], p[1], s=10, label=label, color=colors[labels[ii]])

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
    label = "%s %d" % ('Rank', labels[ii])
    
    ax3.scatter(p[0], p[1], s=10, label=label, color=colors[labels[ii]])

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
    label = "%s %d" % ('Rank', labels[ii])

    ax4.scatter(p[0], p[1], s=10, label=label, color=colors[labels[ii]])

#######################################

# ax1.set_yticks(np.linspace(.8, .98, 7))
ax3.set_yticks([.45, .47, .49, .51])

ax2.set_xticks(range(1, 10+1, 1))
ax4.set_xticks(range(1, 10+1, 1))

ax2.set_xlabel(xlabel='Sparsity', fontsize=10.)
ax4.set_xlabel(xlabel='Sparsity', fontsize=10.)

ax1.set_ylabel(ylabel='Accuracy', fontsize=10.)
ax2.set_ylabel(ylabel='Angle', fontsize=10.)
# ax3.set_ylabel(ylabel='Accuracy')
# ax4.set_ylabel(ylabel='Angle')

f.set_size_inches(7., 5.)

for ax in [ax1, ax2, ax3, ax4]:
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(10.) 
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(10.) 

# lgd = ax4.legend(loc='upper left', bbox_to_anchor=(1.02, 1.5), fontsize=10)

f.subplots_adjust(hspace=0)
# plt.show()
f.savefig('plot2-1.png', dpi=1000)


