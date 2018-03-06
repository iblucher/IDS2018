import numpy as np
from pca import pca

data = np.loadtxt('murderdata2d.txt')
pca(data)
