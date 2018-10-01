
import numpy as np
import matplotlib.cm as cmap
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

weights = np.load('./cifar10/conv1_dfa_0_sparse_0.npy')
# weights = np.load('./cifar10/conv1_dfa_1_sparse_0.npy')

weights = np.transpose(weights)
print (np.shape(weights))

for ii in range(96):
    for jj in range(3):
        if jj == 0:
            row = weights[ii][jj]
        else:
            row = np.concatenate((row, weights[ii][jj]), axis=1)
            
    if ii == 0:
        img = row
    else:
        img = np.concatenate((img, row), axis=0)
  
plt.imsave("img.png", img, cmap="gray")
