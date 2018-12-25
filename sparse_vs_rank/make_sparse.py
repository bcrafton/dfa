
import numpy as np

###################################################

to_sparse = {1: 90, 2:80, 3:70, 4:60, 5:50, 6:40, 7:30, 8:20, 9:10, 10:0}    

###################################################

new_points = []
points = np.load('mnist_data.npy')
for p in points:
    new_p = p
    new_p['sparse'] = to_sparse[p['sparse']]
    new_points.append(new_p)
    
np.save('mnist_data_sparse.npy', new_points)

###################################################

new_points = []
points = np.load('cifar10_data.npy')
for p in points:
    new_p = p
    new_p['sparse'] = to_sparse[p['sparse']]
    new_points.append(new_p)
    
np.save('cifar10_data_sparse.npy', new_points)

###################################################
