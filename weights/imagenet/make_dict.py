
import numpy as np

keys = ['conv1', 'conv1_bias', \
        'conv2', 'conv2_bias', \
        'conv3', 'conv3_bias', \
        'conv4', 'conv4_bias', \
        'conv5', 'conv5_bias']

new_weights = {}           
for key in keys:
    weights = np.load(key + '.npy')
    new_weights[key] = weights
np.save("bp_weights", new_weights)

new_weights = {}           
for key in keys:
    weights = np.load(key + '_dfa.npy')
    new_weights[key] = weights
np.save("dfa_weights", new_weights)
