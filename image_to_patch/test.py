import numpy as np

count = 1
x = np.zeros(shape=(27, 27, 3))
for ii in range(27):
    for jj in range(27):
        for kk in range(3):
            x[ii][jj][kk] = count 
            count = count + 1
            
y = np.copy(x)
y = np.reshape(y, (9, 3, 27, 3))
y = np.transpose(y, (0, 2, 1, 3))
y = np.reshape(y, (81, 3, 3, 3))
y = np.transpose(y, (0, 2, 1, 3))

for ii in range(9):
    for jj in range(9):
        print (y[ii * 9 + jj])
        print (x[ii*3:(ii+1)*3, jj*3:(jj+1)*3, :])
        assert(np.all(y[ii * 9 + jj] == x[ii*3:(ii+1)*3, jj*3:(jj+1)*3, :]))
