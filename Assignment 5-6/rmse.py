from __future__ import division
import numpy as np

def rmse(y, t):
    l = len(y)
    s = 0
    for i in range(l):
        s = s + (np.linalg.norm(y[i] - t[i]))**2
    rmse = 1/l * s
    return np.sqrt(rmse)
