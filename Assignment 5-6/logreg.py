from __future__ import division
import numpy as np

def logreg(data, w):

    # initialize variables
    tol = 1.0e-4
    max_iter = 1.0e5
    lrate = 0.5
    it = 0
    grad = 1

    # separate labels from dataset
    x = data[:, :-1]
    y = data[:, -1]
    r, c = x.shape

    # insert column of 1s into x matrix
    x = np.c_[np.ones(r), x]

    # algorithm loop
    while np.linalg.norm(grad) > tol and it < max_iter:
        # compute gradient
        s = 0
        for i in range(r):
            num = y[i] * x[i, :]
            den = 1 + np.exp(y[i] * np.dot(x[i, :], w))
            s = s + num/den
        grad = -1/r * s

        # step direction
        w = w - lrate * grad
        it = it + 1
        #print(grad)
        #print(np.linalg.norm(grad))
        print(w)

    return w
