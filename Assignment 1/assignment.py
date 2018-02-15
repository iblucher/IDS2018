# main module for assigment 1
import numpy as np
import matplotlib.pyplot as plt
import time
from meanFEV1 import meanFEV1
from hyptest import hyptest
from corr import corr

# Read the dataset from file
data = np.loadtxt('smoking.txt')
data = np.array(data)

# Store dimensions of dataset
[row, col] = np.shape(data)

# Separate datset in smokers and non-smokers
nonsmokers = []
smokers = []
for i in range(row):
    if data[i, 4] == 0:
        nonsmokers.append(data[i, :])
    else:
        smokers.append(data[i, :])
smokers = np.array(smokers)
nonsmokers = np.array(nonsmokers)

# Calculate means of FEV1 scores for both groups
(smean, nmean) = meanFEV1(smokers, nonsmokers)
print("Exercise 1:\n- Smokers mean FEV1 score = %g \n- Non-smokers mean FEV1 score = %g\n" % (smean, nmean))

# Plot boxplot of the two FEV1 scores
labels = ['Smokers', 'Non-smokers']
plt.boxplot([smokers[:, 1], nonsmokers[:, 1]], labels = labels)
plt.title('Boxplot of FEV1 scores for smokers and non smokers')
plt.ylabel('FEV1 score')
plt.show()

# Make hypothesis test on the means
sig = 0.05
t = hyptest(smokers, nonsmokers, sig)
print("True if we can't reject the null hypothesis. False if we can reject the null hypothesis")
print("Result = %r\n" % t)

# Plot bars representing age versus FEV1 scores
plt.bar(data[:, 0], data[:, 1])
plt.xlabel('Age')
plt.ylabel('FEV1 score')
plt.title('FEV1 score for ages 3 to 19')
plt.xlim(2, 20)
plt.show()

# Compute correlation between age and FEV1 scores
r = corr(data[:, 0], data[:, 1])
print("Exercise 4\n- correlation = %g\n" % r)

# Plot histograms over the age of subjects in each groups
plt.hist(smokers[:, 0], edgecolor = 'k')
plt.xlabel('Age of the subjects')
plt.ylabel('Count')
plt.title('Histogram over age of smokers')
plt.show()

plt.hist(nonsmokers[:, 0], edgecolor = 'k')
plt.xlabel('Age of the subjects')
plt.ylabel('Count')
plt.title('Histogram over age of non-smokers')
plt.show()
