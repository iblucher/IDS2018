# data is the datamatrix of the smoking dataset, e.g. as obtained by data = numpy.loadtxt('smoking.txt')
# should return True if the null hypothesis is rejected and False otherwise, i.e. return p < 0.05
import math
import numpy as np
from scipy.stats import ttest_ind, ttest_ind_from_stats, t

def hyptest(smokers, nonsmokers, sig):
	# Compute sample means and variances for smokers and non-smokers
	[srow, scol] = np.shape(smokers)
	[nrow, ncol] = np.shape(nonsmokers)
	smean = smokers[:, 1].mean()
	nmean = nonsmokers[:, 1].mean()
	svar = smokers[:, 1].var(ddof = 1)
	nvar = nonsmokers[:, 1].var(ddof = 1)

	# Compute the two-sample t-statistic
	tt = (smean - nmean) / np.sqrt(svar/srow + nvar/nrow)

	# Compute the degrees of freedom
	freedom = (svar/srow + nvar/nrow)**2 / (svar**2/(srow**2*srow - 1) +  nvar**2/(nrow**2*nrow - 1))
	freedom = math.floor(freedom)
	print(freedom)

	# Compute p-value
	p = 2 * t.cdf(-tt, freedom)

	print("t = %g  p = %g" % (tt, p))

	# Decide if null hypothesis can be rejected
	if(p > sig):
		return True
	return False
