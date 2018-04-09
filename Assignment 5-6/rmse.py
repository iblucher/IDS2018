import numpy as np

def rmse(x, y):
    # insert column of 1s into x matrix
    r = len(x)
    x = np.c_[np.ones(r), x]

    t = np.dot(x, w)

    
