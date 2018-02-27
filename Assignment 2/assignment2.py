import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, zero_one_loss
from sklearn.model_selection import KFold
from sklearn import preprocessing
from checkPerformance import checkPerformance

# read training and test datasets
dataTrain = np.loadtxt('IDSWeedCropTrain.csv', delimiter = ',')
dataTest = np.loadtxt('IDSWeedCropTest.csv', delimiter = ',')

# separate data into x and y subgroups (inputs and outputs)
xTrain = dataTrain[:, :-1]
yTrain = dataTrain[:, -1]
xTest = dataTest[:, :-1]
yTest = dataTest[:, -1]

# INput argument que seleciona se quer normalizar os dados ou nao

# Exercise 1: nearest neighbor classification
nnAccScore = checkPerformance(xTrain, yTrain, xTest, yTest, 1)
print(nnAccScore)

# data normalization
scaler = preprocessing.StandardScaler().fit(xTrain)
xTrainN = scaler.transform(xTrain)
xTestN = scaler.transform(xTest)

xTrain = xTrainN
xTest = xTestN

# cross-validation and hyperparameter selection
avg_loss = []
for i in [1, 3, 5, 7, 9, 11]:
    knn = KNeighborsClassifier(n_neighbors = i)

    cv = KFold(n_splits = 5)
    loss = []
    for train_index, test_index in cv.split(xTrain):
        xTrainCV, xTestCV, yTrainCV, yTestCV = xTrain[train_index], xTrain[test_index], yTrain[train_index], yTrain[test_index]
        knn.fit(xTrainCV, yTrainCV).score(xTestCV, yTestCV)
        loss.append(zero_one_loss(yTestCV, knn.predict(xTestCV)))

    loss = np.array(loss)
    avg_loss.append(np.mean(loss, dtype = np.float64))

avg_loss = np.array(avg_loss)
print(avg_loss)

knnAccScore = checkPerformance(xTrainN, yTrain, xTestN, yTest, 3)
print(knnAccScore)
