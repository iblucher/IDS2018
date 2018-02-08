import numpy as np
import matplotlib.pyplot as plt
import meanFEV1 as mf

data = np.loadtxt('smoking.txt')

mf.meanFEV1(data)
