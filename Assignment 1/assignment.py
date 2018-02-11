import numpy as np
import matplotlib.pyplot as plt
import meanFEV1 as mf

data = np.loadtxt('smoking.txt')

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

(smean, nmean) = mf.meanFEV1(smokers, nonsmokers)

# Boxplot of the two FEV1 scores
labels = ['Smokers', 'Non-smokers']
plt.boxplot([smokers[:, 1], nonsmokers[:, 1]], labels = labels)
plt.title('Boxplot of FEV1 scores for smokers and non smokers')
plt.ylabel('FEV1 score')
plt.show()
