
import numpy as np

weights = np.load("vgg16_weights.npz")

new_keys = ['conv1', 'conv1_bias',   \
            'conv2', 'conv2_bias',   \
            'conv3', 'conv3_bias',   \
            'conv4', 'conv4_bias',   \
            'conv5', 'conv5_bias',   \
            'conv6', 'conv6_bias',   \
            'conv7', 'conv7_bias',   \
            'conv8', 'conv8_bias',   \
            'conv9', 'conv9_bias',   \
            'conv10', 'conv10_bias', \
            'fc1', 'fc1_bias',       \
            'fc2', 'fc2_bias',       \
            'fc3', 'fc3_bias']

keys = ['conv1_1_W', 'conv1_1_b', \
        'conv1_2_W', 'conv1_2_b', \
        'conv2_1_W', 'conv2_1_b', \
        'conv2_2_W', 'conv2_2_b', \
        'conv3_1_W', 'conv3_1_b', \
        'conv3_2_W', 'conv3_2_b', \
        'conv3_3_W', 'conv3_3_b', \
        'conv4_1_W', 'conv4_1_b', \
        'conv4_2_W', 'conv4_2_b', \
        'conv4_3_W', 'conv4_3_b', \
        'conv5_1_W', 'conv5_1_b', \
        'conv5_2_W', 'conv5_2_b', \
        'conv5_3_W', 'conv5_3_b', \
        'fc6_W', 'fc6_b',         \
        'fc7_W', 'fc7_b',         \
        'fc8_W', 'fc8_b']

new_weights = {}
for ii in range(len(keys)):
    key = keys[ii]
    new_key = new_keys[ii]
    new_weights[key] = weights[key]
    
np.save("vgg", new_weights)

