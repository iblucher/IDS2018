import numpy as np

def reshape_centers(centers):
    c1 = np.reshape(centers[0, ], (28, 28))
    c2 = np.reshape(centers[1, ], (28, 28))
    c3 = np.reshape(centers[2, ], (28, 28))
    return c1, c2, c3
