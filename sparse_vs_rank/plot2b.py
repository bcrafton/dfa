
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

colors = {5:  '#C86464', \
          10: '#2020FF'}

#######################################

plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['font.size'] = 10.

f, ax3 = plt.subplots(1, 1)

#######################################

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

#######################################

cifar10 = np.load('cifar10_data_sparse.npy')
mnist = np.load('mnist_data_sparse.npy')

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
    key = (d['rank'], d['sparse'])
    
    if key in data_grouped.keys():
        data_grouped[key].append(d['acc'] * 100.)
    else:
        data_grouped[key] = [d['acc'] * 100.]
        
points = []
labels = [] 
for key in data_grouped:
    points.append( (np.average(data_grouped[key]), np.std(data_grouped[key])) )
    labels.append( key )
'''
for ii in range(len(points)):
    p = points[ii]
    ax1.errorbar(x=labels[ii][1], y=p[0], yerr=p[1], fmt='o', color=colors[labels[ii][0]])
'''
#######################################
'''
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
    key = (d['rank'], d['sparse'])
    
    if key in data_grouped.keys():
        data_grouped[key].append(d['angle'])
    else:
        data_grouped[key] = [d['angle']]
        
points = []
labels = [] 

for key in data_grouped:
    points.append( (np.average(data_grouped[key]), np.std(data_grouped[key])) )
    labels.append( key )

for ii in range(len(points)):
    p = points[ii]
    ax2.errorbar(x=labels[ii][1], y=p[0], yerr=p[1], fmt='o', color=colors[labels[ii][0]])
'''
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
    key = (d['rank'], d['sparse'])
    
    if key in data_grouped.keys():
        data_grouped[key].append(d['acc'] * 100.)
    else:
        data_grouped[key] = [d['acc'] * 100.]
        
points = []
labels = [] 
for key in sorted(data_grouped):
    print (key)
    if key == (10, 90):
        # print (key)
        # points.append( (50., np.std(data_grouped[key])) )
        points.append( (np.average(data_grouped[key]), np.std(data_grouped[key])) )
    else:
        points.append( (np.average(data_grouped[key]), np.std(data_grouped[key])) )
    labels.append( key )

for ii in range(len(points)):
    p = points[ii]
    ax3.errorbar(x=labels[ii][1], y=p[0], yerr=p[1], fmt='o', color=colors[labels[ii][0]])

#######################################
'''
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
    key = (d['rank'], d['sparse'])
    
    if key in data_grouped.keys():
        data_grouped[key].append(d['angle'])
    else:
        data_grouped[key] = [d['angle']]
        
points = []
labels = [] 
for key in data_grouped:
    points.append( (np.average(data_grouped[key]), np.std(data_grouped[key])) )
    labels.append( key )

for ii in range(len(points)):
    p = points[ii]
    ax4.errorbar(x=labels[ii][1], y=p[0], yerr=p[1], fmt='o', color=colors[labels[ii][0]])
'''
#######################################
'''
ax1.set_ylim(96.6, 98.)
ax1.set_yticks([96.8, 97.0, 97.2, 97.4, 97.6, 97.8])
'''
'''
ax2.set_ylim(40, 54)
ax2.set_yticks([42, 44, 46, 48, 50, 52])
'''
ax3.set_ylim(40, 54)
ax3.set_yticks([42, 44, 46, 48, 50, 52])
'''
ax4.set_ylim(45, 85)
ax4.set_yticks([50, 55, 60, 65, 70, 75, 80])
'''
#######################################

# ax1.set_xticks(range(0, 100, 10))
ax3.set_xticks(range(0, 100, 10))

# ax1.set_xlim(95, -5)
# ax1.invert_xaxis()
ax3.set_xlim(95, -5)
# ax3.invert_xaxis()

# ax2.set_xlabel(xlabel='Sparsity (%)', fontsize=10.)
# ax4.set_xlabel(xlabel='Sparsity (%)', fontsize=10.)

# ax1.set_ylabel(ylabel='Accuracy (%)', fontsize=10.)
# ax2.set_ylabel(ylabel='Angle (Degrees)', fontsize=10.)
ax3.set_ylabel(ylabel='Accuracy (%)', fontsize=10.)
# ax4.set_ylabel(ylabel='Angle')

f.set_size_inches(3.5, 2.5)

for ax in [ax3]:
    ax.grid(alpha=0.9, linestyle='--', linewidth=0.35)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(10.) 
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(10.) 

# lgd = ax4.legend(loc='upper left', bbox_to_anchor=(1.02, 1.5), fontsize=10)

f.subplots_adjust(hspace=0)
# plt.show()
f.savefig('plot2.png', bbox_inches='tight', dpi=1000)


