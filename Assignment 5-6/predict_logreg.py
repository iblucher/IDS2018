import numpy as np

def predict_logreg(x, w):

    r, c = x.shape
    y = np.dot(x, w)

    for i in range(r):
        if y[i] >= 0:
            y[i] = 1
        else:
            y[i] = -1

    return y
