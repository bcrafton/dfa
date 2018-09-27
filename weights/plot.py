
import numpy as np
import matplotlib.cm as cmap
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

weights = np.load('conv1_dfa_1_sparse_1.npy')

weights = np.transpose(weights)
print (np.shape(weights))

# weights = weights[0][0]                
# plt.imsave('weights.png', weights, cmap=cmap.get_cmap('gray'))

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
