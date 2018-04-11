from __future__ import division
import numpy as np

def compute_percentages(labels, clusters):

    # counter vector to find percentage of labels per cluster
    counter = np.zeros((3, 3))
    for i in range(clusters.shape[0]):
        if labels[i] == 1:
            counter[clusters[i], 0] += 1
        elif labels[i] == 7:
            counter[clusters[i], 1] += 1
        elif labels[i] == 9:
            counter[clusters[i], 2] += 1

    print(counter)
    total = np.sum(counter, axis = 1)
    print(total)

    percentages = np.zeros((3, 3))
    for i in range(3):
        percentages[i, 0] = counter[i, 0] / total[i]
        percentages[i, 1] = counter[i, 1] / total[i]
        percentages[i, 2] = counter[i, 2] / total[i]

    return percentages
