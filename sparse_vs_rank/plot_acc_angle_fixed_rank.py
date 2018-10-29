
import numpy as np
import math
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.cm as cmap

#######################################

# fig = plt.figure(figsize=(10., 10.))

f, [[ax1, ax3], [ax2, ax4]] = plt.subplots(2, 2, sharex=True, sharey=False)

#######################################

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

#######################################

def get_data():
    data = []

    itrs = range(1, 10+1)

    for sparse in range(1, 10+1):
        for rank in range(1, 10+1):
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
                
    return data
#######################################

data = get_data()

ii = 0
while ii < len(data):
    d = data[ii]
    if d['rank'] != 5:
        del data[ii]
    else:
        ii += 1

data_grouped = {}
ii = 0
for ii in range(len(data)):
    d = data[ii]
    key = d['sparse']
    
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
    label = "%s %d" % ('sparse', labels[ii])
    
    ax1.scatter(p[0], p[1], s=10, label=label)

#######################################

data = get_data()

ii = 0
while ii < len(data):
    d = data[ii]
    if d['rank'] != 10:
        del data[ii]
    else:
        ii += 1

data_grouped = {}
ii = 0
for ii in range(len(data)):
    d = data[ii]
    key = d['sparse']
    
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
    label = "%s %d" % ('sparse', labels[ii])
    
    ax1.scatter(p[0], p[1], s=10, label=label)

#######################################

f.subplots_adjust(hspace=0)

#f.savefig('samplefigure', bbox_extra_artists=(lgd,), bbox_inches='tight')
f.savefig('samplefigure', bbox_inches='tight')




