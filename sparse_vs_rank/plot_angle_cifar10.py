
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
sparses = [0, 1]

all_accs = []
all_angles = []

for sparse in sparses:
    accs = []
    angles = []

    for itr in itrs:
        fname = "./cifar10_dfa/sparse%ditr%d.npy" % (sparse, itr)
        results = np.load(fname).item()
        
        acc = np.max(results['acc'])
        accs.append(acc)
      
        fb = results['fc3_fb']
        w2 = results['fc4']
        fb = np.reshape(fb.T, (-1))
        w2 = np.reshape(w2, (-1))
        angle = angle_between(fb, w2) * (180. / 3.14)
        angles.append(angle)
        
        print ("sparse %d itr %d acc %f angle %f" % (sparse, itr, acc, angle))

    scatter = plt.scatter(angles, accs, s=10, label="Sparse " + str(sparse))

    all_accs.extend(accs)
    all_angles.extend(angles)
     
##########################

fit = np.poly1d(np.polyfit(all_angles, all_accs, 1))
pred_xs = all_angles
pred_ys = fit(pred_xs)
plt.plot(pred_xs, pred_ys)

##########################

plt.xlabel("Angle", fontsize=18)
plt.xticks(fontsize=14)

plt.ylabel("Accuracy", fontsize=18)
plt.yticks(fontsize=14)

plt.legend(fontsize=18, markerscale=4.0)
plt.savefig('angle_vs_acc.png')

