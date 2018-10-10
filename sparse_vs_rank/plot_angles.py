
import numpy as np
import math
from scipy import stats
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--benchmark', type=str, default='mnist')
parser.add_argument('--itrs', type=int, default=10)
parser.add_argument('--rank', type=int, default=10)
args = parser.parse_args()

#######################################

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

#######################################

sparses = []
ranks =[]
accs = []
angles = []

itrs = range(1, args.itrs+1)

for sparse in range(1, args.rank + 1): 
    for itr in itrs:
    
        fname = args.benchmark + "/sparse%drank%ditr%d.npy" % (sparse, args.rank, itr)
        results = np.load(fname).item()
        
        fb = results['fc3_fb']
        w2 = results['fc4']
        acc = np.max(results['acc'])

        # dont forget transpose
        fb = np.reshape(fb.T, (-1))
        w2 = np.reshape(w2, (-1))
        angle = angle_between(fb, w2) * (180. / 3.14)

        sparses.append(sparse)
        ranks.append(args.rank)
        accs.append(acc)
        angles.append(angle)
        
        print ("sparse %d rank %d itr %d acc %f" % (sparse, args.rank, itr, acc))
    
#######################################

labels = ["Sparse", "Rank", "Accuracy", "Angle"]

idx = 0
xidx = 2
yidx = 3

data = np.array([sparses, ranks, accs, angles])
data = np.transpose(data).tolist()
data = sorted(data, key=lambda tup: tup[idx])

length = len(data)
grouped = {}
for ii in range(length):
    key = data[ii][idx]
    if key not in grouped.keys():
        grouped[key] = [ (data[ii][xidx], data[ii][yidx]) ]
    else:
        grouped[key].append( (data[ii][xidx], data[ii][yidx]) )

print grouped.keys()

fig = plt.figure(figsize=(10, 10))
for key in grouped.keys():
    print (np.shape(grouped[key]))
    x = np.transpose(grouped[key]).tolist()[0]
    y = np.transpose(grouped[key]).tolist()[1]
    scatter = plt.scatter(x, y, s=10, label=labels[idx] + " " + str(int(key)))

#######################################

plt.xlabel(labels[xidx], fontsize=18)
plt.xticks(fontsize=14)

plt.ylabel(labels[yidx], fontsize=18)
plt.yticks(fontsize=14)

plt.legend(fontsize=18, markerscale=4.0)
plt.savefig(args.benchmark + '_acc_vs_angle')

#######################################

    
