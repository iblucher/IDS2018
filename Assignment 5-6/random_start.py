import numpy as np

def random_start(data):
    init1 = data[np.random.choice(len(data))]
    init2 = data[np.random.choice(len(data))]
    init3 = data[np.random.choice(len(data))]

    starting_point = np.vstack((init1, init2, init3))
    return starting_point
