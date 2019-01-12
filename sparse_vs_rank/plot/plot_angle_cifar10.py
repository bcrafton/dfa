
import numpy as np
import matplotlib.pyplot as plt
import argparse

# parser = argparse.ArgumentParser()
# args = parser.parse_args()

#######################################################

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))
    
#######################################################

fig = plt.figure(figsize=(10, 10))

#######################################################

itrs = range(50)
# sparses = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
# sparses = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
sparses = [0]

all_accs = []
all_angles = []

def compute_angles(fbs, ws):
    assert(len(ws) == len(fbs)+1)
    angles = []
    num = len(ws)
    
    start = num - 2
    end = -1
    
    for ii in range(start, end, -1):
        if ii == start:
            w = ws[ii + 1]
        else:
            w = np.dot(ws[ii + 1], w)
        
        _fb = np.reshape(fbs[ii].T, (-1))
        _w = np.reshape(w, (-1))
        angle = angle_between(_fb, _w) * (180. / 3.14)
        angles.append(angle)
    
    return angles
        
for sparse in sparses:
    accs = []
    angles = []

    for itr in itrs:
        fname = "./cifar10_dfa/sparse%ditr%d.npy" % (sparse, itr)
        results = np.load(fname).item()
        
        acc = np.max(results['acc'])
        accs.append(acc)

        fb1 = results['fc1_fb']
        fb2 = results['fc2_fb']
        fb3 = results['fc3_fb']

        w1 = results['fc1']
        w2 = results['fc2']
        w3 = results['fc3']
        w4 = results['fc4']
        
        ret = compute_angles([fb1, fb2, fb3], [w1, w2, w3, w4])
        angle = np.average(ret)
        angles.append(angle)
        
        print ("sparse %d itr %d acc %f angle %f" % (sparse, itr, acc, angle))

    plt.scatter(angles, accs, s=10, label="Sparse " + str(sparse))

    all_accs.append(accs)
    all_angles.append(angles)

all_angles_flat = np.reshape(all_angles, (-1))
all_accs_flat = np.reshape(all_accs, (-1))

##########################

fit = np.poly1d(np.polyfit(all_angles_flat, all_accs_flat, 1))
pred_xs = all_angles_flat
pred_ys = fit(pred_xs)
plt.plot(pred_xs, pred_ys)

##########################

plt.xlabel("Angle", fontsize=18)
plt.xticks(fontsize=14)

plt.ylabel("Accuracy", fontsize=18)
plt.yticks(fontsize=14)

plt.legend(fontsize=18, markerscale=4.0)
plt.savefig('angle_vs_acc.png')

##########################
