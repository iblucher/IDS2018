from __future__ import division
import numpy as np

def gradient_descent():

    tol = 1.0e-10
    max_iter = 1.0e4
    x = 1
    lrate = 0.0001
    it = 0

    grad = 20*x - np.exp(-x/2) * 0.5
    print(np.linalg.norm(grad))

    while np.linalg.norm(grad) > tol and it < max_iter:
        grad = 20*x - np.exp(-x/2) * 0.5
        x = x - lrate * grad
        it = it + 1

    f = np.exp(-x/2) + 10 * (x ** 2)
    return f, it
