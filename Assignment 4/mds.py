# input:   1) datamatrix as loaded by numpy.loadtxt('dataset.txt')
#	   2) an integer d specifying the number of dimensions for the output (most commonly used are 2 or 3)
# output:  1) an N x d numpy array containing the d coordinates of the N original datapoints projected onto the top d PCs
import numpy as np
from pca import pca

def mds(data, d):
    (evals, evecs, mean) = pca(data)
    ev = evecs[:, 0:d]
    return(data.dot(ev))
