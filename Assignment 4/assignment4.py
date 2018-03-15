import numpy as np
import matplotlib.pyplot as plt

data_diatoms = np.loadtxt('diatoms.txt')
r, c = data_diatoms.shape

# Exercise 1
# plot single cell by plotting landmark points and interpolating between subsequent points
diatoms_x = []
diatoms_y = []
for i in range(0, c - 1, 2):
    diatoms_x.append(data_diatoms[1, i])
    diatoms_y.append(data_diatoms[1, i + 1])
plt.scatter(diatoms_x, diatoms_y, marker = '.')
plt.plot(diatoms_x, diatoms_y)
plt.axis('equal')
plt.show()

# plot all cells on top of each other
for i in range(r):
    diatoms_x = []
    diatoms_y = []
    for j in range(0, c - 1, 2):
        diatoms_x.append(data_diatoms[i, j])
        diatoms_y.append(data_diatoms[i, j + 1])
    plt.scatter(diatoms_x, diatoms_y, marker = '.')
    plt.plot(diatoms_x, diatoms_y)
plt.axis('equal')
plt.show()
