# input: datamatrix as loaded by numpy.loadtxt('dataset.txt')
# output:  1) the eigenvalues in a vector (numpy array) in descending order
#          2) the unit eigenvectors in a matrix (numpy array) with each column being an eigenvector (in the same order as its associated eigenvalue)
#
# note: make sure the order of the eigenvalues (the projected variance) is decreasing, and the eigenvectors have the same order as their associated eigenvalues
import numpy as np
import numpy.matlib as nmat

def pca(data):
    [M, N] = data.shape

    # center data
    mean = np.mean(data, axis = 1)
    for i in range(M):
        data[i, :] = data[i, :] - mean[i]
    #print(data)

    # find covariance matrix
    cov = np.matmul(data, data.transpose())
    cov = np.array(cov)
    cov = 1 / (N - 1) * cov
    print(cov)

    
