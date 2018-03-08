# input: datamatrix as loaded by numpy.loadtxt('dataset.txt')
# output:  1) the eigenvalues in a vector (numpy array) in descending order
#          2) the unit eigenvectors in a matrix (numpy array) with each column being an eigenvector (in the same order as its associated eigenvalue)
#
# note: make sure the order of the eigenvalues (the projected variance) is decreasing, and the eigenvectors have the same order as their associated eigenvalues
from __future__ import division
import numpy as np
from sklearn.preprocessing import StandardScaler

def pca(data):
    # M stands for number of dimensions and N stands for number of trials
    data = data.transpose()
    [M, N] = data.shape
    print(M)
    print(N)

    # center data
    mean = np.mean(data, axis = 1)
    #print(mean)
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
    evals = np.array(evals)
    #print(evals)
    #print(evecs)

    # confirm eigenvectors have unit length 1
    for ev in evecs:
        np.testing.assert_array_almost_equal(1.0, np.linalg.norm(ev))

    # sorting eigenvectors by decreasing eigenvalues
    eig_pairs = []
    for i in range(len(evals)):
        eig_pairs.append((evals[i], evecs[:, i]))
    eig_pairs.sort(key = lambda x: x[0], reverse = True)

    evals_sorted = []
    evecs_sorted = []
    for i in range(len(evals)):
        evals_sorted.append(eig_pairs[i][0])
        evecs_sorted.append(eig_pairs[i][1])
    evals_sorted = np.array(evals_sorted)
    evecs_sorted = np.array(evecs_sorted)
    # this transposition is so that the eigenvectors are the column and not the row vectors of the evecs_sorted matrix
    evecs_sorted = evecs_sorted.transpose()

    #print(evals_sorted)
    #print(evecs_sorted)

    return (evals_sorted, evecs_sorted)
