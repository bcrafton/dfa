
import numpy as np

###################################################

to_sparse = {1:10, 2:20, 3:30, 4:40, 5:50, 6:60, 7:70, 8:80, 9:90, 10:100}    

###################################################

new_points = []
points = np.load('mnist_data.npy')
for p in points:
    new_p = p
    new_p['sparse'] = to_sparse[p['sparse']]
    new_points.append(new_p)
    
np.save('mnist_data_connect.npy', new_points)

###################################################

new_points = []
points = np.load('cifar10_data.npy')
for p in points:
    new_p = p
    new_p['sparse'] = to_sparse[p['sparse']]
    new_points.append(new_p)
    
np.save('cifar10_data_connect.npy', new_points)

###################################################
