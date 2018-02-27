from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def checkPerformance(xTrain, yTrain, xTest, yTest, k):
    # initialize nearest neighbor classifier and fit the training data
    nn = KNeighborsClassifier(n_neighbors = k)
    nn.fit(xTrain, yTrain)

    accTest = accuracy_score(yTest, nn.predict(xTest))
    return(accTest)
