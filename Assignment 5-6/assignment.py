import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import accuracy_score, zero_one_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from multivarlinreg import multivarlinreg
from predict_linreg import predict_linreg
from rmse import rmse
from gradient_descent import gradient_descent
from plot_iris import plot_iris
from transform_labels import transform_labels
from logreg import logreg
from predict_logreg import predict_logreg
from compute_percentages import compute_percentages

##############
# Exercise 2 #
##############
# read in red wine dataset
wine_train = np.loadtxt('redwine_training.txt')
wine_test = np.loadtxt('redwine_testing.txt')

x_wine_train = wine_train[:, :-1]
y_wine_train = wine_train[:, -1]
x_wine_test = wine_test[:, :-1]
y_wine_test = wine_test[:, -1]

# exercise 2b
w_2b = multivarlinreg(x_wine_train[:, 0], y_wine_train)
print('Exercise 2b: weight vector for only first feature')
print(w_2b)
print('\n')

# exercise 2c
w_2c = multivarlinreg(x_wine_train, y_wine_train)
print('Exercise 2c: weigth vector for all features')
print(w_2c)
print('\n')

##############
# Exercise 3 #
##############
t_3b = predict_linreg(x_wine_test[:, 0], w_2b)
rm_3b = rmse(y_wine_test, t_3b)
print('Exercise 3b: RMSE score for only feature')
print(rm_3b)
print('\n')

t_3c = predict_linreg(x_wine_test, w_2c)
rm_3c = rmse(y_wine_test, t_3c)
print('Exercise 3c: RMSE score for all features')
print(rm_3c)
print('\n')

##############
# Exercise 5 #
##############
# read in crop dataset
crop_train = np.loadtxt('IDSWeedCropTrain.csv', delimiter = ',')
crop_test = np.loadtxt('IDSWeedCropTest.csv', delimiter = ',')

x_crop_train = crop_train[:, :-1]
y_crop_train = crop_train[:, -1]
x_crop_test = crop_test[:, :-1]
y_crop_test = crop_test[:, -1]

rfc = RandomForestClassifier(n_estimators = 50)
rfc.fit(x_crop_train, y_crop_train)
accTest = accuracy_score(y_crop_test, rfc.predict(x_crop_test))
#print(accTest)

##############
# Exercise 6 #
##############
#gradient_descent()

##############
# Exercise 7 #
##############
iris2d1_train = np.loadtxt('Iris2D1_train.txt')
iris2d1_test = np.loadtxt('Iris2D1_test.txt')
iris2d2_train = np.loadtxt('Iris2D2_train.txt')
iris2d2_test = np.loadtxt('Iris2D2_test.txt')

iris2d1 = np.append(iris2d1_train, iris2d1_test, axis = 0)
iris2d2 = np.append(iris2d2_train, iris2d2_test, axis = 0)

# plot iris datasets with color coded points
#plot_iris(iris2d1, 1)
#plot_iris(iris2d2, 2)

# transform all zero labels into -1 so slide algorithm works
y_iris2d1_train = transform_labels(iris2d1_train[:, -1])
y_iris2d1_test = transform_labels(iris2d1_test[:, -1])
y_iris2d2_train = transform_labels(iris2d2_train[:, -1])
y_iris2d2_test = transform_labels(iris2d2_test[:, -1])

# separate x matrix from dataset
x_iris2d1_train = iris2d1_train[:, :-1]
x_iris2d1_test = iris2d1_test[:, :-1]
x_iris2d2_train = iris2d2_train[:, :-1]
x_iris2d2_test = iris2d2_test[:, :-1]

# insert column of 1s into x matrices
x_iris2d1_train = np.c_[np.ones(x_iris2d1_train.shape[0]), x_iris2d1_train]
x_iris2d1_test = np.c_[np.ones(x_iris2d1_test.shape[0]), x_iris2d1_test]
x_iris2d2_train = np.c_[np.ones(x_iris2d2_train.shape[0]), x_iris2d2_train]
x_iris2d2_test = np.c_[np.ones(x_iris2d2_test.shape[0]), x_iris2d2_test]

#w2d1 = logreg(x_iris2d1_train, y_iris2d1_train, np.zeros(iris2d1_train.shape[1]))
#pred_2d1 = predict_logreg(x_iris2d1_test, w2d1)
#loss_2d1 = zero_one_loss(y_iris2d1_test, pred_2d1)
#print(loss_2d1)


#w2d2 = logreg(x_iris2d2_train, y_iris2d2_train, np.zeros(iris2d2_train.shape[1]))
#pred_2d2 = predict_logreg(x_iris2d2_test, w2d2)
#loss_2d2 = zero_one_loss(y_iris2d2_test, pred_2d2)
#print(loss_2d2)

##############
# Exercise 9 #
##############
mnist_digits = np.loadtxt('MNIST_179_digits.txt')
mnist_labels = np.loadtxt('MNIST_179_labels.txt')

# figure out how to initialize starting point
starting_point = np.vstack((mnist_digits[23, ], mnist_digits[394, ], mnist_digits[638, ]))
kmeans = KMeans(n_clusters = 3, init = starting_point, algorithm = 'full').fit(mnist_digits)
print(kmeans.labels_)
print(kmeans.labels_.shape)

p = compute_percentages(mnist_labels, kmeans.labels_)
print(p)

centers = kmeans.cluster_centers_

c1 = np.resize(centers[0, ], (28, 28))
c2 = np.resize(centers[1, ], (28, 28))
c3 = np.resize(centers[2, ], (28, 28))

# create images from cluster centers
im1 = Image.fromarray(c1.astype('uint8'), mode = 'L')
im2 = Image.fromarray(c2.astype('uint8'), mode = 'L')
im3 = Image.fromarray(c3.astype('uint8'), mode = 'L')
im1.save('im1.png')
im2.save('im2.png')
im3.save('im3.png')

xTrain = mnist_digits[0:900, ]
yTrain = mnist_labels[0:900, ]
xTest = mnist_digits[900:1125, ]
yTest = mnist_labels[900:1125, ]

# k neighbors classifier (exercise 9b)
avg_loss = []
for i in [1, 3, 5, 7, 9]:
    knn = KNeighborsClassifier(n_neighbors = i)
    # initialize cross-validation
    cv = KFold(n_splits = 5)
    loss = []
    for train_index, test_index in cv.split(xTrain):
        xTrainCV, xTestCV, yTrainCV, yTestCV = xTrain[train_index], xTrain[test_index], yTrain[train_index], yTrain[test_index]
        knn.fit(xTrainCV, yTrainCV)
        loss.append(zero_one_loss(yTestCV, knn.predict(xTestCV)))
    loss = np.array(loss)
    avg_loss.append(np.mean(loss, dtype = np.float64))
print(avg_loss)

nn = KNeighborsClassifier(n_neighbors = 1)
nn.fit(xTrain, yTrain)
accK = accuracy_score(yTest, nn.predict(xTest))
print(accK)

###############
# Exercise 10 #
###############

# use PCA on MNIST dataset

# plot cumulativa variance

# run clustering with k = 3 (on projected data with 20 and 200 dimensions)

# count percentages for both dimensions

# knn classifier and n fold validation for both dimensions
