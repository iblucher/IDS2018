# x and y should be vectors of equal length
# should return their correlation as a number
from scipy.stats import pearsonr

def corr(x,y):
	# Calculate Pearson's correlation coefficient
	(r, p) = pearsonr(x, y)
	return r
