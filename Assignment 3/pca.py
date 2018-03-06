# input: datamatrix as loaded by numpy.loadtxt('dataset.txt')
# output:  1) the eigenvalues in a vector (numpy array) in descending order
#          2) the unit eigenvectors in a matrix (numpy array) with each column being an eigenvector (in the same order as its associated eigenvalue)
#
# note: make sure the order of the eigenvalues (the projected variance) is decreasing, and the eigenvectors have the same order as their associated eigenvalues
from __future__ import division
import numpy as np

def pca(data):
    data = data.transpose()
    [M, N] = data.shape

    # center data
    mean = np.mean(data, axis = 0)
    for i in range(M):
        data[i, :] = data[i, :] - mean[i]
    #print(data)

    # find covariance matrix
    datat = data.transpose()
    cov = np.dot(data, datat)
    cov = np.array(cov)
    cov_mat = 1 / N * cov
    #print(cov_mat)

    # find eigenvectors and eigenvalues
    (evals, evecs) = np.linalg.eig(cov_mat)
    print(evals)
    print(evecs)

    # confirm eigenvectors have unit length 1
    for ev in evecs:
        np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))

    # sorting eigenvectors by decreasing eigenvalues
