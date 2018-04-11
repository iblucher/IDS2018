from __future__ import division
import numpy as np

def logreg(x, y, w):

    # initialize variables
    tol = 1.0e-4
    max_iter = 1.0e5
    lrate = 0.5
    it = 0
    grad = 1

    r, c = x.shape

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

    return w
