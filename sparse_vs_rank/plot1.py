
import numpy as np
import math
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
import argparse

#######################################

parser = argparse.ArgumentParser()
parser.add_argument('--markers', type=int, default=0)
args = parser.parse_args()

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

'''
markers = {1:  's', \
           2:  'P', \
           3:  '*', \
           4:  '<', \
          
           5:  'o', \
           6:  '+', \
           7:  'X', \
           8:  'D', \
          
           9:  'v', \
           10: '^'}
'''

if args.markers:
    markers = {1:  'x', \
               2:  'o', \
               3:  'o', \
               4:  'o', \
              
               5:  'o', \
               6:  'o', \
               7:  'o', \
               8:  'o', \
              
               9:  'o', \
               10: '+'}
               
    sizes   = {1:  28, \
               2:  10, \
               3:  10, \
               4:  10, \
              
               5:  10, \
               6:  10, \
               7:  10, \
               8:  10, \
              
               9:  10, \
               10: 28}

else:
    markers = {1:  'o', \
               2:  'o', \
               3:  'o', \
               4:  'o', \
              
               5:  'o', \
               6:  'o', \
               7:  'o', \
               8:  'o', \
              
               9:  'o', \
               10: 'o'}

    sizes   = {1:  10, \
               2:  10, \
               3:  10, \
               4:  10, \
              
               5:  10, \
               6:  10, \
               7:  10, \
               8:  10, \
              
               9:  10, \
               10: 10}

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
    key = (d['rank'], d['sparse'])
    
    if key in data_grouped.keys():
        data_grouped[key].append(d['acc'] * 100.)
    else:
        data_grouped[key] = [d['acc'] * 100.]
        
points1 = []
labels1 = [] 
for key in data_grouped:    
    points1.append( np.average(data_grouped[key]) )
    labels1.append( key )

#######################################

data_grouped = {}
ii = 0
for ii in range(len(mnist)):
    d = mnist[ii]
    key = (d['rank'], d['sparse'])
    
    if key in data_grouped.keys():
        data_grouped[key].append(d['angle'])
    else:
        data_grouped[key] = [d['angle']]
        
points2 = []
labels2 = [] 
for key in data_grouped:    
    points2.append( np.average(data_grouped[key]) )
    labels2.append( key )

#######################################

data_grouped = {}
ii = 0
for ii in range(len(cifar10)):
    d = cifar10[ii]
    key = (d['rank'], d['sparse'])
    
    if key in data_grouped.keys():
        data_grouped[key].append(d['acc'] * 100.)
    else:
        data_grouped[key] = [d['acc'] * 100.]
        
points3 = []
labels3 = [] 
for key in data_grouped:    
    points3.append( np.average(data_grouped[key]) )
    labels3.append( key )

#######################################

data_grouped = {}
ii = 0
for ii in range(len(cifar10)):
    d = cifar10[ii]
    key = (d['rank'], d['sparse'])
    
    if key in data_grouped.keys():
        data_grouped[key].append(d['angle'])
    else:
        data_grouped[key] = [d['angle']]
        
points4 = []
labels4 = [] 
for key in data_grouped:    
    points4.append( np.average(data_grouped[key]) )
    labels4.append( key )
    
#######################################

plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['font.size'] = 10

f, [ax1, ax3] = plt.subplots(1, 2)

for ii in range(len(points1)):
    p = points1[ii]
    ax1.scatter(labels1[ii][0], p, s=sizes[labels1[ii][1]], color=colors[labels1[ii][1]], marker=markers[labels1[ii][1]])

for ii in range(len(points2)):
    p = points2[ii]
    # ax2.scatter(labels2[ii][0], p, s=sizes[labels1[ii][1]], color=colors[labels2[ii][1]], marker=markers[labels1[ii][1]])

for ii in range(len(points3)):
    p = points3[ii]
    ax3.scatter(labels3[ii][0], p, s=sizes[labels1[ii][1]], color=colors[labels3[ii][1]], marker=markers[labels1[ii][1]])

for ii in range(len(points4)):
    p = points4[ii]
    # ax4.scatter(labels4[ii][0], p, s=sizes[labels1[ii][1]], color=colors[labels4[ii][1]], marker=markers[labels1[ii][1]])

# ax2.set_xticks(range(1, 11))
# ax4.set_xticks(range(1, 11))

# ax1.set_yticks(np.linspace(.8, .98, 7))

plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['font.size'] = 10.

'''
ax1.set_yticks(np.linspace(95, 98, 4))
ax1.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

ax3.set_yticks(np.linspace(30, 50, 5))
ax3.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
'''

ax1.set_xticks(range(1, 10+1, 1))
ax3.set_xticks(range(1, 10+1, 1))
'''
ax2.set_xlabel(xlabel='Rank', fontsize=10)
ax4.set_xlabel(xlabel='Rank', fontsize=10)
'''
ax1.set_ylabel(ylabel='Accuracy (%)', fontsize=10)
# ax2.set_ylabel(ylabel='Angle (' + u'\N{DEGREE SIGN}' + ')', fontsize=10)
# ax2.set_ylabel(ylabel='Angle (Degrees)', fontsize=10)

ax1.set_ylim(94.5, 98.)
ax1.set_yticks([95., 95.5, 96., 96.5, 97., 97.5])
'''
ax2.set_ylim(40, 54)
ax2.set_yticks([42, 44, 46, 48, 50, 52])
'''
ax3.set_ylim(25, 55)
ax3.set_yticks([30, 35, 40, 45, 50])
# ax3.set_yticks([30, 34, 38, 42, 46, 50])
'''
ax4.set_ylim(50, 90)
ax4.set_yticks([55, 60, 65, 70, 75, 80, 85])
'''
# ax3.set_ylabel(ylabel='Accuracy')
# ax4.set_ylabel(ylabel='Angle')

# if we turn the ticks off
# f.subplots_adjust(hspace=0.05)

# otherwise no space
f.subplots_adjust(hspace=0.0)

f.set_size_inches(7., 3.)

# lgd = ax4.legend(loc='upper left', bbox_to_anchor=(1.02, 1.5), fontsize=10)

for ax in [ax1, ax3]:
    ax.grid(alpha=0.9, linestyle='--', linewidth=0.35)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(10) 
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(10) 

# f.savefig('plot1-' + str(args.markers) + '.png', bbox_inches='tight', dpi=300)
f.savefig('plot1.png', bbox_inches='tight', dpi=300)



