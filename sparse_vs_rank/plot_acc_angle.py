
import numpy as np
import math
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.cm as cmap
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--benchmark', type=str, default='mnist')
parser.add_argument('--itrs', type=int, default=10)

parser.add_argument('--fix_key', type=str, default=None)
parser.add_argument('--fix_val', type=int, default=0)

parser.add_argument('--group_key', type=str, default='sparse')

parser.add_argument('--color_key', type=str, default=None)

parser.add_argument('--x', type=str, default=None)
parser.add_argument('--y', type=str, default=None)

args = parser.parse_args()

#######################################

# fig = plt.figure(figsize=(10., 10.))

#######################################

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

#######################################

data = []

itrs = range(1, args.itrs+1)

for sparse in range(1, 10+1):
    for rank in range(sparse, 10+1):
        for itr in itrs:
        
            fname = args.benchmark + "/sparse%drank%ditr%d.npy" % (sparse, rank, itr)
            results = np.load(fname).item()
            
            if args.benchmark == 'mnist':
                fb = results['fc1_fb']
                w2 = results['fc2']
            else:
                fb = results['fc3_fb']
                w2 = results['fc4']
            
            acc = np.max(results['acc'])

            # dont forget transpose
            fb = np.reshape(fb.T, (-1))
            w2 = np.reshape(w2, (-1))
            angle = angle_between(fb, w2) * (180. / 3.14)
            
            data.append({"sparse":sparse, "rank":rank, "acc":acc, "angle":angle})

#######################################

data_grouped = {}
ii = 0
for ii in range(len(data)):
    d = data[ii]
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
for ii in range(len(data)):
    d = data[ii]
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

f, [[ax1, ax3], [ax2, ax4]] = plt.subplots(2, 2, sharex=True, sharey=False)
for ii in range(len(points1)):
    p = points1[ii]
    p = np.transpose(p)
    label = "%s %d" % (args.group_key, labels1[ii])
    ax1.scatter(p[0], p[1], s=10, label=label)

for ii in range(len(points2)):
    p = points2[ii]
    p = np.transpose(p)
    label = "%s %d" % (args.group_key, labels2[ii])
    ax2.scatter(p[0], p[1], s=10, label=label)

f.subplots_adjust(hspace=0)

plt.show()





