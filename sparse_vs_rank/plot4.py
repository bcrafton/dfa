
import numpy as np
import math
from scipy import stats
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--benchmark', type=str, default='mnist')
parser.add_argument('--itrs', type=int, default=10)

parser.add_argument('--fix_key', type=str, default=None)
parser.add_argument('--fix_val', type=int, default=0)

parser.add_argument('--group_key', type=str, default=None)

parser.add_argument('--x', type=str, default=None)
parser.add_argument('--y', type=str, default=None)

args = parser.parse_args()

#######################################

fig = plt.figure(figsize=(10, 10))

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
            
            print ("sparse %d rank %d itr %d acc %f angle %f" % (sparse, rank, itr, acc, angle))
    
#######################################

assert(args.x)

#######################################

assert(args.y)

#######################################

if args.fix_key:
    ii = 0
    while ii < len(data):
        d = data[ii]
        if d[args.fix_key] != args.fix_val:
            del data[ii]
        else:
            ii += 1

#######################################

if args.group_key:
    data_grouped = {}
    ii = 0
    for ii in range(len(data)):
        d = data[ii]
        key = d[args.group_key]
        
        if key in data_grouped.keys():
            data_grouped[key].append([d[args.x], d[args.y]])
        else:
            data_grouped[key] = [[d[args.x], d[args.y]]]

else:
    data_grouped = {}
    ii = 0
    for ii in range(len(data)):
        d = data[ii]
        key = ''
        
        if key in data_grouped.keys():
            data_grouped[key].append([d[args.x], d[args.y]])
        else:
            data_grouped[key] = [[d[args.x], d[args.y]]]

points = []
labels = [] 
for key in data_grouped:
    points.append( data_grouped[key] )
    labels.append( key )

#######################################

if args.group_key:
    dim = 3
else:
    dim = 2

if dim == 3:
    for ii in range(len(points)):
        p = points[ii]
        p = np.transpose(p)
        label = "%s %d" % (args.group_key, labels[ii])
        plt.scatter(p[0], p[1], s=30, label=label)

    if args.fix_key:
        name = "%s_%s_%d_%s_%s_%s" % (args.benchmark, args.fix_key, args.fix_val, args.x, args.y, args.group_key)
    else:
        name = "%s_%s_%s_%s" % (args.benchmark, args.x, args.y, args.group_key)

    plt.xlabel(args.x, fontsize=18)
    plt.xticks(fontsize=14)

    plt.ylabel(args.y, fontsize=18)
    plt.yticks(fontsize=14)
    
    plt.legend(fontsize=18, markerscale=2.0)

    if args.fix_key:
        title = "%s %s %d" % (args.benchmark, args.fix_key, args.fix_val)
    else:
        title = args.benchmark
    plt.title(title, fontsize=18)

    plt.savefig(name + '.png')

else:
    points = np.transpose(points)
    plt.scatter(points[0], points[1], s=30)
    
    if args.fix_key:
        name = "%s_%s_%d_%s_%s" % (args.benchmark, args.fix_key, args.fix_val, args.x, args.y)
    else:
        name = "%s_%s_%s_%s" % (args.benchmark, args.x, args.y)

    plt.xlabel(args.x, fontsize=18)
    plt.xticks(fontsize=14)

    plt.ylabel(args.y, fontsize=18)
    plt.yticks(fontsize=14)

    if args.fix_key:
        title = "%s %s %d" % (args.benchmark, args.fix_key, args.fix_val)
    else:
        title = args.benchmark
    plt.title(title, fontsize=18)

    plt.savefig(name + '.png')
    
    
    
    
    
    
    
    
    
    
