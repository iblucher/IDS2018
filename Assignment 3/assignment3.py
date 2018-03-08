import numpy as np
import matplotlib.pyplot as plt
from pca import pca

murder_data = np.loadtxt('murderdata2d.txt')

(evals, evecs) = pca(murder_data)

# scatter plot for murder dataset
# x is the percent unemployed and y is the murders per annum per 1000000 inhabitants
x = murder_data[:, 0]
y = murder_data[:, 1]
#print(x)
#print(y)
plt.scatter(x,y)
plt.title("Scatter plot of the murder data")
plt.xlabel("Percent unemployed")
plt.ylabel("Murders per annum per 1,000,000 inhabitants")
plt.axis('equal')

s0 = np.sqrt(evals[0])
s1 = np.sqrt(evals[1])
plt.plot([0, s0*evecs[0,0]], [0, s0*evecs[1,0]], 'r')
plt.plot([0, s1*evecs[0,1]], [0, s1*evecs[1,1]], 'r')

plt.show()



# scatter plot for crop dataset
