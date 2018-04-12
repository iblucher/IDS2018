from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent():

    tol = 1.0e-10
    max_iter = 1.0e4
    x = 1
    lrate = 0.0001
    it = 0

    grad = 20*x - 0.5 * np.exp(-x/2)

    while np.linalg.norm(grad) > tol and it < max_iter:
        grad = 20*x - 0.5 * np.exp(-x/2)
        #print(grad)
        x = x - lrate * grad
        it = it + 1

    f = np.exp(-x/2) + 10 * (x ** 2)
    return f, it

f, it = gradient_descent()
