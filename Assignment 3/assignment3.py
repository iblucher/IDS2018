import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from pca import pca
from mds import mds

# Exercise 1
# load murder dataset
murder_data = np.loadtxt('murderdata2d.txt')

# perform pca on murder dataset
(evals_murder, evecs_murder) = pca(murder_data)

# scatter plot for murder dataset
# x is the percent unemployed and y is the murders per annum per 1000000 inhabitants
x = murder_data[:, 0]
y = murder_data[:, 1]
plt.scatter(x,y)
plt.title("Scatter plot of the centered murder data")
plt.xlabel("Percent unemployed")
plt.ylabel("Murders per annum per 1,000,000 inhabitants")
plt.axis('equal')

# scale the eigenvectors and plot the mean point
s0 = np.sqrt(evals_murder[0])
s1 = np.sqrt(evals_murder[1])
plt.plot(0, 0, marker = 'x', markersize = 7, color='k', label = 'Mean', linestyle = 'None')
plt.plot([0, s0*evecs_murder[0,0]], [0, s0*evecs_murder[1,0]], 'r', label = 'Eigenvectors')
plt.plot([0, s1*evecs_murder[0,1]], [0, s1*evecs_murder[1,1]], 'r')
plt.axis('equal')
plt.legend()
plt.show()
plt.close()

# scatter plot for crop dataset
pest_data_train = np.loadtxt('IDSWeedCropTrain.csv', delimiter = ',')
pest_data_test = np.loadtxt('IDSWeedCropTest.csv', delimiter = ',')
pest_x_train = pest_data_train[:, :-1]
pest_y_train = pest_data_train[:, -1]
pest_x_test = pest_data_test[:, :-1]
pest_y_test = pest_data_test[:, -1]

(evals_pest, evecs_pest) = pca(pest_x_train)

plt.plot(evals_pest)
plt.xlabel('PCs index')
plt.ylabel('Projected variance')
plt.title('Variance versus principal component index in descending order')
plt.show()
plt.close()

# cumulative variance plot
total = np.sum(evals_pest)
cumulative_var = []
s = 0
for ev in evals_pest:
    s = s + ev/total
    cumulative_var.append(s)
plt.plot(cumulative_var)
plt.title('Cumulative variance versus number of used PCs')
plt.xlabel('Used PCs index')
plt.ylabel('Cumulative variance')
plt.show()
plt.close()

# Exercise 2
mds_scaled = mds(pest_x_train, 2)
x = mds_scaled[:, 0]
y = mds_scaled[:, 1]
plt.scatter(x, y)
plt.title('Pesticide dataset projected onto the first 2 PCs')
plt.xlabel('Data projected onto PC 1')
plt.ylabel('Data projected onto PC 2')
plt.show()
plt.close()

# Exercise 3
sp = np.vstack((pest_x_train[0, ], pest_x_train[1, ]))
kmeans = KMeans(n_clusters = 2, algorithm = 'full', n_init = 1, init = sp).fit(pest_x_train)
print('Exercise 3: clustering')
print(kmeans.cluster_centers_)
