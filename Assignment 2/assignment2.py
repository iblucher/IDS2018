import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# read training and test datasets
dataTrain = np.loadtxt('IDSWeedCropTrain.csv', delimiter = ',')
dataTest = np.loadtxt('IDSWeedCropTest.csv', delimiter = ',')

# separate data into x and y subgroups (inputs and outputs)
xTrain = dataTrain[:, :-1]
yTrain = dataTrain[:, -1]
xTest = dataTest[:, :-1]
yTest = dataTest[:, -1]

# initialize nearest neighbor classifier and fit the training data
nn = KNeighborsClassifier(n_neighbors = 1)
nn.fit(xTrain, yTrain)

accTest = accuracy_score(yTest, nn.predict(xTest))
print(accTest)
