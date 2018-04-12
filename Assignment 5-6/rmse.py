from __future__ import division
import numpy as np

def rmse(y, t):
    l = len(y)
    s = 0
    for i in range(l):
        a = (y[i] - t[i])**2
        s += a
    rmse = s / l
    return np.sqrt(rmse)
