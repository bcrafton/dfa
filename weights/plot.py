
import numpy as np
import matplotlib.cm as cmap
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

weights = np.load('imagenet/conv1_dfa.npy')
weights = np.transpose(weights)
weights = np.reshape(weights, (16, 18, 11, 11))

for ii in range(16):
    for jj in range(18):
        if jj == 0:
            row = weights[ii][jj]
        else:
            row = np.concatenate((row, weights[ii][jj]), axis=1)
            
    if ii == 0:
        img = row
    else:
        img = np.concatenate((img, row), axis=0)
  
plt.imsave("alexnet_dfa.png", img, cmap="gray")
