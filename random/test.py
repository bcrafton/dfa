
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import scipy
import scipy.stats as st
import statsmodels as sm

np.set_printoptions(threshold=np.inf)
x = np.load('cifar10_fc_weights.npy').item()

####################################

print ('========================')

fc2 = x['fc2']
fc3 = x['fc3']
fc4 = x['fc4']

fc1_fb = np.dot(fc2, np.dot(fc3, fc4))
fc2_fb = np.dot(fc3, fc4)
fc3_fb = fc4

print (np.shape(fc1_fb), np.std(fc1_fb), np.average(fc1_fb))
print (np.shape(fc2_fb), np.std(fc2_fb), np.average(fc2_fb))
print (np.shape(fc3_fb), np.std(fc3_fb), np.average(fc3_fb))
print (np.shape(fc2), np.std(fc2), np.average(fc2))
print (np.shape(fc3), np.std(fc3), np.average(fc3))
print (np.shape(fc4), np.std(fc4), np.average(fc4))

####################################

print ('========================')

sqrt_fan_out = np.sqrt(1000) # will always be 1000 for this network
high = 1.0 / sqrt_fan_out
low = -high

# sizes = [1000, 1000, 1000, 10]
a = np.random.uniform(low, high, size=(1000, 1000))
b = np.random.uniform(low, high, size=(1000, 1000))
c = np.random.uniform(low, high, size=(1000, 10))

abc = np.dot(a, np.dot(b, c))
bc = np.dot(b, c)
c = c

print ('fc1_fb', np.shape(abc), np.std(abc), np.average(abc))
print ('fc2_fb', np.shape(bc), np.std(bc), np.average(bc))
print ('fc3_fb', np.shape(c), np.std(c), np.average(c))
print ('fc2', np.shape(a), np.std(a), np.average(a))
print ('fc3', np.shape(b), np.std(b), np.average(b))
print ('fc4', np.shape(c), np.std(c), np.average(c))

####################################

print ('========================')

a = np.random.normal(loc=np.mean(fc2), scale=np.std(fc2), size=(1000, 1000))
b = np.random.normal(loc=np.mean(fc3), scale=np.std(fc3), size=(1000, 1000))
c = np.random.normal(loc=np.mean(fc4), scale=np.std(fc4), size=(1000, 10))

abc = np.dot(a, np.dot(b, c))
bc = np.dot(b, c)
c = c

print ('fc1_fb', np.shape(abc), np.std(abc), np.average(abc))
print ('fc2_fb', np.shape(bc), np.std(bc), np.average(bc))
print ('fc3_fb', np.shape(c), np.std(c), np.average(c))
print ('fc2', np.shape(a), np.std(a), np.average(a))
print ('fc3', np.shape(b), np.std(b), np.average(b))
print ('fc4', np.shape(c), np.std(c), np.average(c))

####################################
'''
y = np.reshape(fc2, (-1))
size = len(y)
x = scipy.arange(size)

dist_names = ['gamma', 'beta', 'rayleigh', 'norm', 'pareto']
for dist_name in dist_names:
    dist = getattr(scipy.stats, dist_name)
    param = dist.fit(y)
    pdf_fitted = dist.pdf(x, *param[:-2], loc=param[-2], scale=param[-1]) * size
    plt.plot(pdf_fitted, label=dist_name)
    plt.xlim(0,47)
    
plt.legend(loc='upper right')
plt.show()
'''
####################################

print ('========================')

params = st.gennorm.fit(np.reshape(fc2, (-1)))
arg = params[:-2]
loc = params[-2]
scale = params[-1]
a = st.gennorm.rvs(size=np.shape(fc2), loc=loc, scale=scale, *arg)

params = st.gennorm.fit(np.reshape(fc3, (-1)))
arg = params[:-2]
loc = params[-2]
scale = params[-1]
b = st.gennorm.rvs(size=np.shape(fc3), loc=loc, scale=scale, *arg)

params = st.gennorm.fit(np.reshape(fc4, (-1)))
arg = params[:-2]
loc = params[-2]
scale = params[-1]
c = st.gennorm.rvs(size=np.shape(fc4), loc=loc, scale=scale, *arg)

abc = np.dot(a, np.dot(b, c))
bc = np.dot(b, c)
c = c

print ('fc1_fb', np.shape(abc), np.std(abc), np.average(abc))
print ('fc2_fb', np.shape(bc), np.std(bc), np.average(bc))
print ('fc3_fb', np.shape(c), np.std(c), np.average(c))
print ('fc2', np.shape(a), np.std(a), np.average(a))
print ('fc3', np.shape(b), np.std(b), np.average(b))
print ('fc4', np.shape(c), np.std(c), np.average(c))

####################################

# plt.hist(np.reshape(fc1_fb, (-1)), bins=100)
# plt.hist(np.reshape(abc, (-1)), bins=100)

# plt.hist(np.reshape(fc2_fb, (-1)), bins=100)
# plt.hist(np.reshape(b, (-1)), bins=100)

# plt.hist(np.reshape(fc3_fb, (-1)), bins=100)
# plt.hist(np.reshape(c, (-1)), bins=100)

# plt.show()





















