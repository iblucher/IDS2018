# data is the datamatrix of the smoking dataset, e.g. as obtained by data = numpy.loadtxt('smoking.txt')
# should return a tuple containing average FEV1 of smokers and nonsmokers

import numpy as np

def meanFEV1(data):
	[row, col] = np.shape(data)
	print(row)
	print(col)

	# SEPARAR O DATASET ENTRE FUMANTES E NAO-FUMANTES (quinta coluna)
	nonsmokers = []
	smokers = []
	for i in range(row):
	    if data[i, 4] == 0:
	        nonsmokers.append(data[i, :])
	    else:
	        smokers.append(data[i, :])
	smokers = np.array(smokers)
	nonsmokers = np.array(nonsmokers)

	# CALCULAR O FEV1 MEDIO PRA CADA MATRIZ
	[sr, sc] = np.shape(smokers)
	[nr, nc]  = np.shape(nonsmokers)
	smean = smokers[:, 1].mean()
	nmean = nonsmokers[:, 1].mean()
	print(smean)
	print(nmean)
	
