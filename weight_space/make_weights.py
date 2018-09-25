
import numpy as np

LAYER1 = 784
LAYER2 = 100
LAYER3 = 10

sqrt_fan_in = np.sqrt(LAYER1)
high = 1.0 / sqrt_fan_in
low = -high
weights1 = np.random.uniform(low=low, high=high, size=(LAYER1, LAYER2))

sqrt_fan_in = np.sqrt(LAYER2)
high = 1.0 / sqrt_fan_in
low = -high
weights2 = np.random.uniform(low=low, high=high, size=(LAYER2, LAYER3))

np.save("W1_init", weights1)
np.save("W2_init", weights2)
