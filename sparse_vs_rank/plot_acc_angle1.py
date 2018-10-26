
import numpy as np
import math
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.cm as cmap

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

itrs = range(1, 10+1)

for sparse in range(1, 10+1):
    for rank in range(sparse, 10+1):
        for itr in itrs:
        
            fname = "mnist/sparse%drank%ditr%d.npy" % (sparse, rank, itr)
            results = np.load(fname).item()
            
            fb = results['fc1_fb']
            w2 = results['fc2']
            
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

data = []

itrs = range(1, 10+1)

for sparse in range(1, 10+1):
    for rank in range(sparse, 10+1):
        for itr in itrs:
        
            fname = "cifar10/sparse%drank%ditr%d.npy" % (sparse, rank, itr)
            results = np.load(fname).item()
            
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
        
points3 = []
labels3 = [] 
for key in data_grouped:
    points3.append( data_grouped[key] )
    labels3.append( key )

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
        
points4 = []
labels4 = [] 
for key in data_grouped:
    points4.append( data_grouped[key] )
    labels4.append( key )

#######################################

f, [[ax1, ax3], [ax2, ax4]] = plt.subplots(2, 2, sharex=True, sharey=False)

for ii in range(len(points1)):
    p = points1[ii]
    p = np.transpose(p)
    label = "%s %d" % ('sparse', labels1[ii])
    ax1.scatter(p[0], p[1], s=10, label=label)

for ii in range(len(points2)):
    p = points2[ii]
    p = np.transpose(p)
    label = "%s %d" % ('sparse', labels2[ii])
    ax2.scatter(p[0], p[1], s=10, label=label)

for ii in range(len(points3)):
    p = points3[ii]
    p = np.transpose(p)
    label = "%s %d" % ('sparse', labels3[ii])
    ax3.scatter(p[0], p[1], s=10, label=label)

for ii in range(len(points4)):
    p = points4[ii]
    p = np.transpose(p)
    label = "%s %d" % ('sparse', labels4[ii])
    ax4.scatter(p[0], p[1], s=10, label=label)

f.subplots_adjust(hspace=0)

plt.savefig('lol.png')





