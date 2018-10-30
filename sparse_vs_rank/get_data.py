
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

def get_data(benchmark):
    data = []

    itrs = range(1, 10+1)

    for sparse in range(1, 10+1):
        for rank in range(1, 10+1):
            if sparse * rank > 10:
                for itr in itrs:
                    fname = benchmark + "/sparse%drank%ditr%d.npy" % (sparse, rank, itr)
                    results = np.load(fname).item()
                    
                    if benchmark == 'cifar10':
                        fb = results['fc3_fb']
                        w2 = results['fc4']
                    else:
                        fb = results['fc1_fb']
                        w2 = results['fc2']
                    
                    acc = np.max(results['acc'])

                    # dont forget transpose
                    fb = np.reshape(fb.T, (-1))
                    w2 = np.reshape(w2, (-1))
                    angle = angle_between(fb, w2) * (180. / 3.14)
                    
                    data.append({"sparse":sparse, "rank":rank, "acc":acc, "angle":angle})
                
    return data
    
#######################################

#data = get_data('cifar10')
#np.save('cifar10_data', data)

#######################################

data = get_data('mnist')
np.save('mnist_data', data)

#######################################


