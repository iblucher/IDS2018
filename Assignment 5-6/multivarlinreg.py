import numpy as np

def multivarlinreg(x, y):

    # insert column of 1s into x matrix
    r = len(x)
    x = np.c_[np.ones(r), x]

    # analitically compute weight vector w
    xt = np.transpose(x)
    xtx = np.linalg.inv(np.dot(xt, x))
    w = np.dot(xtx, xt)
    w = np.dot(w, y)

    return w
