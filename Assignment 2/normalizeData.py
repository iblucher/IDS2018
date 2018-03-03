from sklearn import preprocessing

def normalizeData(xTrain, xTest):
    scaler = preprocessing.StandardScaler().fit(xTrain)
    # data normalization
    xTrainN = scaler.transform(xTrain)
    xTestN = scaler.transform(xTest)

    xTrain = xTrainN
    xTest = xTestN

    return (xTrainN, xTestN)
