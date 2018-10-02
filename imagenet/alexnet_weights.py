
import numpy as np

weights = np.load(open("bvlc_alexnet.npy", "rb"), encoding="latin1").item()
keys = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']

print (weights.keys())

new_weights = {}
for key in keys:
    w, b = weights[key]
    print (np.shape(w), np.shape(b))
    
    new_weights[key] = w
    new_weights[key + '_bias'] = b

# np.save("alexnet_weights", new_weights)

####################################

'''
weights = np.load("alexnet_weights.npy").item()
keys = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8']

for key in keys:
    w = weights[key]
    b = weights[key + '_bias']
    print (np.shape(w), np.shape(b))
'''
