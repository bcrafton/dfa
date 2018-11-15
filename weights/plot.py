
import numpy as np
import matplotlib.cm as cmap
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

dpi=300

############################################################################

weights = np.load('alexnet_weights.npy').item()
weights = weights['conv1']
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
  
plt.imsave('alexnet_bp.pdf', img, cmap='gray', dpi=dpi)
plt.imsave('alexnet_bp.svg', img, cmap='gray', dpi=dpi)

'''
# requires that you use .eps, not support for .svg
img = Image.open('alexnet_bp.eps')
width = int(17. / 2. * dpi)
height = width
img = img.resize((width,height), Image.ANTIALIAS)
img.save('alexnet_bp.eps') 
'''

############################################################################

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
  
plt.imsave("alexnet_dfa.pdf", img, cmap="gray", dpi=dpi)
plt.imsave("alexnet_dfa.svg", img, cmap="gray", dpi=dpi)

############################################################################
