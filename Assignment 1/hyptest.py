
# data is the datamatrix of the smoking dataset, e.g. as obtained by data = numpy.loadtxt('smoking.txt')
# should return True if the null hypothesis is rejected and False otherwise, i.e. return p < 0.05
import numpy as np
from scipy.stats import ttest_ind, ttest_ind_from_stats

def hyptest(smokers, nonsmokers):
	[srow, scol] = np.shape(smokers)
	[nrow, ncol] = np.shape(nonsmokers)

	(t, p) = ttest_ind(smokers[:, 1], nonsmokers[:, 1], equal_var = False)
	svar = smokers[:, 1].var()
	nvar = nonsmokers[:, 1].var()

	s = svar/srow
	n = nvar/ncol
	den = np.sqrt(s + n)
	tt = (smokers[:, 1].mean() - nonsmokers[:, 1].mean())/den
	print(tt)
	print("ttest_ind:            t = %g  p = %g" % (t, p))
