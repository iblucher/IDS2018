import numpy as np
import matplotlib.pyplot as plt
from pca import pca

murder_data = np.loadtxt('murderdata2d.txt')

(evals_murder, evecs_murder) = pca(murder_data)

# scatter plot for murder dataset
# x is the percent unemployed and y is the murders per annum per 1000000 inhabitants
x = murder_data[:, 0]
y = murder_data[:, 1]
plt.scatter(x,y)
plt.title("Scatter plot of the centered murder data")
plt.xlabel("Percent unemployed")
plt.ylabel("Murders per annum per 1,000,000 inhabitants")
#plt.axis('equal')

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
print(evals_pest)
#print(evecs_pest.shape)

plt.plot(evals_pest)
plt.xlabel('PCs index')
plt.ylabel('Projected variance')
plt.title('Variance versus principal component index in descending order')
plt.show()
