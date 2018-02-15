# data is the datamatrix of the smoking dataset, e.g. as obtained by data = numpy.loadtxt('smoking.txt')
# should return a tuple containing average FEV1 of smokers and nonsmokers
import numpy as np

def meanFEV1(smokers, nonsmokers):
	# Calculate the average FEV1 score for each group
	smean = smokers[:, 1].mean()
	nmean = nonsmokers[:, 1].mean()
	return(smean, nmean)
