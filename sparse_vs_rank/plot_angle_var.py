
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--benchmark', type=str, default='mnist')
parser.add_argument('--itrs', type=int, default=10)
args = parser.parse_args()

fig = plt.figure(figsize=(10, 10))

###############################################################

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

###############################################################

benchmark = args.benchmark
itrs = range(1, args.itrs+1)

accs1 = []
w2s = []

accs2 = []
ranks = []
fbs = []

for itr in range(1, 1000+1, 1):
    fname = "bp/itr%d.npy" % (itr)
    print (fname)
    results = np.load(fname).item()
    acc = np.max(results['acc'])
    w2 = results['fc2']
    w2 = np.reshape(w2, (-1))

    accs1.append(acc)
    w2s.append(w2)

for ii in range(1000):
    w2 = w2s[ii]
    w2 = np.reshape(w2, (-1, 1))
    if ii == 0:
        w2mat = w2
    else:
        w2mat = np.concatenate((w2mat, w2), axis=1)
        
# mean = np.mean(w2mat, axis=0, keepdims=True)
mean = np.mean(w2mat, axis=0)

for ii in range(1000):
    w2s[ii] -= mean

'''
print (np.shape(w2mat))
print (np.shape(w2))
print (np.shape(mean))
assert(False)
'''

for sparse in range(1, 10+1):
    for rank in range(10, 10+1):
        for itr in itrs:
            fname = benchmark + "/sparse%drank%ditr%d.npy" % (sparse, rank, itr)
            print (fname)
            results = np.load(fname).item()
            acc = np.max(results['acc'])
            fb = results['fc1_fb']
            fb = np.reshape(fb.T, (-1))
            fb -= mean

            accs2.append(acc)
            ranks.append(rank)
            fbs.append(fb)

var = []
for fb in fbs:
    angles = []
    for w2 in w2s:
        angle = angle_between(fb, w2) * 180. / np.pi
        angles.append(angle)
        
    var.append(np.std(angles))

scatter = plt.scatter(var, accs2, s=10)

plt.xlabel("Var", fontsize=18)
plt.xticks(fontsize=14)

plt.ylabel("Accuracy", fontsize=18)
plt.yticks(fontsize=14)

# plt.legend(fontsize=18, markerscale=4.0)
plt.show()

