import numpy as np
import matplotlib.pyplot as plt
from pca import pca

data_diatoms = np.loadtxt('diatoms.txt')
r, c = data_diatoms.shape

# Exercise 1
# plot single cell by plotting landmark points and interpolating between subsequent points
diatoms_x = []
diatoms_y = []
for i in range(0, c - 1, 2):
    diatoms_x.append(data_diatoms[0, i])
    diatoms_y.append(data_diatoms[0, i + 1])
plt.scatter(diatoms_x, diatoms_y, marker = '.', color = 'b')
plt.plot(diatoms_x, diatoms_y, 'b')
plt.plot([diatoms_x[0], diatoms_x[89]], [diatoms_y[0], diatoms_y[89]], 'b')
plt.axis('equal')
plt.xlabel('x-coordinates')
plt.ylabel('y-coordinates')
#plt.show()
plt.close()

# plot all cells on top of each other
for i in range(r):
    diatoms_x = []
    diatoms_y = []
    for j in range(0, c - 1, 2):
        diatoms_x.append(data_diatoms[i, j])
        diatoms_y.append(data_diatoms[i, j + 1])
    plt.scatter(diatoms_x, diatoms_y, marker = '.')
    plt.plot(diatoms_x, diatoms_y)
plt.xlabel('x-coordinates')
plt.ylabel('y-coordinates')
plt.axis('equal')
#plt.show()
plt.close()

# Exercise 2
evals, evecs, mean = pca(data_diatoms)
s0 = np.sqrt(evals[0])
s1 = np.sqrt(evals[1])
s2 = np.sqrt(evals[2])
e0 = evecs[:, 0]
e1 = evecs[:, 1]
e2 = evecs[:, 2]
temporal0 = [mean - 2*s0*e0, mean - s0*e0, mean, mean + s0*e0, mean + 2*s0*e0]
temporal1 = [mean - 2*s1*e1, mean - s1*e1, mean, mean + s1*e1, mean + 2*s1*e1]
temporal2 = [mean - 2*s2*e2, mean - s2*e2, mean, mean + s2*e2, mean + 2*s2*e2]

for m in range(5):
    spatial_var_x = []
    spatial_var_y = []
    spatial_var = temporal0[m]
    for i in range(0, c - 1, 2):
        spatial_var_x.append(spatial_var[i])
        spatial_var_y.append(spatial_var[i + 1])
    plt.scatter(spatial_var_x, spatial_var_y, marker = '.')
    plt.plot(spatial_var_x, spatial_var_y)
plt.axis('equal')
plt.show()
