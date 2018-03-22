import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pca import pca
from mds import mds
from visualize_var import visualize_var

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
plt.show()
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
plt.show()
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

# plot spatial variance for the first 3 PCs
visualize_var(temporal0, c)
visualize_var(temporal1, c)
visualize_var(temporal2, c)

# Exercise 3
data_toy = np.loadtxt('pca_toydata.txt')
scaled_toy = mds(data_toy, 2)
scaled_toy_x = scaled_toy[:, 0]
scaled_toy_y = scaled_toy[:, 1]
plt.scatter(scaled_toy_x, scaled_toy_y)
plt.axis('equal')
plt.xlabel('Data projected onto the first PC')
plt.ylabel('Data projected onto the second PC')
plt.show()
plt.close()

# remove two last data points
rt, ct = data_toy.shape
new_data_toy = data_toy[:-2, :]
scaled_new = mds(new_data_toy, 2)
scaled_new_x = scaled_new[:, 0]
scaled_new_y = scaled_new[:, 1]
plt.scatter(scaled_new_x, scaled_new_y)
plt.axis('equal')
plt.xlabel('Data projected onto the first PC')
plt.ylabel('Data projected onto the second PC')
plt.show()
plt.close()

# Exercise 4
data_train = np.loadtxt('IDSWeedCropTrain.csv', delimiter = ',')
data_test = np.loadtxt('IDSWeedCropTest.csv', delimiter = ',')
x_train = data_train[:, :-1]
y_train = data_train[:, -1]
x_test = data_test[:, :-1]
y_test = data_test[:, -1]

# project crop data onto first 2 PCs
scaled_data = mds(x_train, 2)
scaled_x = scaled_data[:, 0]
scaled_y = scaled_data[:, 1]

# plot projected data
for i in range(len(y_train)):
    if y_train[i] == 1:
        c = 'skyblue'
    elif y_train[i] == 0:
        c = 'slateblue'
    plt.scatter(scaled_x[i], scaled_y[i], color = c, marker = '.')

# run kmeans clustering
starting_point = np.vstack((x_train[0, ], x_train[1, ]))
kmeans = KMeans(n_clusters = 2, n_init = 1, init = starting_point, algorithm = 'full').fit(x_train)
centers = kmeans.cluster_centers_
eva, eve, m = pca(x_train)
eve = eve[:, :2]
projected_centers = np.matmul(centers, eve)
print('Exercise 4: projected cluster centers')
print(projected_centers)
plt.scatter(projected_centers[:, 0], projected_centers[:, 1], 75, 'r', 'x')
plt.xlabel('Data projected onto the first PC')
plt.ylabel('Data projected onto the second PC')
plt.axis('equal')
plt.show()
plt.close()
