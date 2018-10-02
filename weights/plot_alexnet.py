
import numpy as np
import matplotlib.cm as cmap
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

weights = np.load('alexnet_weights.npy').item()
weights = weights['conv1']
weights = np.transpose(weights)
# 288 11x11 filters
# 96x3, 32x9, 16x18
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
  
plt.imsave("img.png", img, cmap="gray")
