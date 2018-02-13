import numpy as np
import matplotlib.pyplot as plt
import meanFEV1 as mf
import hyptest as hp

# Read the dataset from file
data = np.loadtxt('smoking.txt')

# Store dimensions of dataset
[row, col] = np.shape(data)
print(row)
print(col)

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

# Store dimensions of subgroups
[srow, scol] = np.shape(smokers)
[nrow, ncol] = np.shape(nonsmokers)
print(srow)
print(nrow)

# Calculate means of FEV1 scores for both groups
(smean, nmean) = mf.meanFEV1(smokers, nonsmokers)

# Plot boxplot of the two FEV1 scores
labels = ['Smokers', 'Non-smokers']
plt.boxplot([smokers[:, 1], nonsmokers[:, 1]], labels = labels)
plt.title('Boxplot of FEV1 scores for smokers and non smokers')
plt.ylabel('FEV1 score')
plt.show()

# Make hypothesis test on the means
hp.hyptest(smokers, nonsmokers)
