import sys
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, zero_one_loss
from sklearn.model_selection import KFold
from checkPerformance import checkPerformance
from normalizeData import normalizeData

a = 0
if(str(sys.argv[1]) == '1'):
    a = 1

# read training and test datasets
dataTrain = np.loadtxt('IDSWeedCropTrain.csv', delimiter = ',')
dataTest = np.loadtxt('IDSWeedCropTest.csv', delimiter = ',')

# separate data into x and y subgroups (inputs and outputs)
xTrain = dataTrain[:, :-1]
yTrain = dataTrain[:, -1]
xTest = dataTest[:, :-1]
yTest = dataTest[:, -1]

# Exercise 1: nearest neighbor classification
nnAccScore = checkPerformance(xTrain, yTrain, xTest, yTest, 1)
print('Exercise 1\n1-NN accuracy score = %g\n' % nnAccScore)

# Exercise 4: data normalization
if(a == 1):
    (xTrain, xTest) = normalizeData(xTrain, xTest)

# Exercise 2: cross-validation and hyperparameter selection
avg_loss = []
for i in [1, 3, 5, 7, 9, 11]:
    knn = KNeighborsClassifier(n_neighbors = i)
    # initialize 5-fold cross-validation
    cv = KFold(n_splits = 5)
    loss = []
    for train_index, test_index in cv.split(xTrain):
        xTrainCV, xTestCV, yTrainCV, yTestCV = xTrain[train_index], xTrain[test_index], yTrain[train_index], yTrain[test_index]
        knn.fit(xTrainCV, yTrainCV).score(xTestCV, yTestCV)
        loss.append(zero_one_loss(yTestCV, knn.predict(xTestCV)))
    loss = np.array(loss)
    avg_loss.append(np.mean(loss, dtype = np.float64))

# Exercise 3: evaluation of classification performance for k_best
avg_loss = np.array(avg_loss)
print('Average 0-1 loss for k values in [1, 3, 5, 7, 9, 11]')
print(avg_loss)
print('\n')

knnAccScore = checkPerformance(xTrain, yTrain, xTest, yTest, 3)
print('Exercise 3\nk_best-NN accuracy score = %g\n' % knnAccScore)
