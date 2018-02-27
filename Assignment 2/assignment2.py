import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, zero_one_loss
from sklearn.model_selection import KFold

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
#print(accTest)

# cross-validation and hyperparameter selection
knn = KNeighborsClassifier(n_neighbors = 1)

cv = KFold(n_splits = 5)
loss = []
for train_index, test_index in cv.split(xTrain):
    xTrainCV, xTestCV, yTrainCV, yTestCV = xTrain[train_index], xTrain[test_index], yTrain[train_index], yTrain[test_index]
    print(knn.fit(xTrainCV, yTrainCV).score(xTestCV, yTestCV))
    loss.append(zero_one_loss(yTestCV, knn.predict(xTestCV)))

loss = np.array(loss)
print(loss)

avg_loss = np.mean(loss, dtype = np.float64)
print(avg_loss)
