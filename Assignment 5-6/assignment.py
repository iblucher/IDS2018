# Python packages imports
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, zero_one_loss
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold

# Created functions for the assignment
from multivarlinreg import multivarlinreg
from predict_linreg import predict_linreg
from rmse import rmse
from gradient_descent import gradient_descent
from plot_iris import plot_iris
from transform_labels import transform_labels
from logreg import logreg
from predict_logreg import predict_logreg
from compute_percentages import compute_percentages
from pca import pca
from mds import mds
from random_start import random_start
from reshape_centers import reshape_centers
from draw_images import draw_images

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
#print('Exercise 2b: weight vector for only first feature')
#print(w_2b)
#print('\n')

# exercise 2c
w_2c = multivarlinreg(x_wine_train, y_wine_train)
#print('Exercise 2c: weigth vector for all features')
#print(w_2c)
#print('\n')

##############
# Exercise 3 #r
##############
t_3b = predict_linreg(x_wine_test[:, 0], w_2b)
rm_3b = rmse(y_wine_test, t_3b)
#print('Exercise 3b: RMSE score for only feature')
#print(rm_3b)
#print('\n')

t_3c = predict_linreg(x_wine_test, w_2c)
rm_3c = rmse(y_wine_test, t_3c)
#print('Exercise 3c: RMSE score for all features')
#print(rm_3c)
#print('\n')

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
accTestRF = accuracy_score(y_crop_test, rfc.predict(x_crop_test))
#print('Exercise 5: accuracy score for Random Forest classifier')
#print(accTestRF)
#print('\n')

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
#print(w2d1)
#pred_2d1 = predict_logreg(x_iris2d1_test, w2d1)
#loss_2d1 = zero_one_loss(y_iris2d1_test, pred_2d1)
#print(loss_2d1)


#w2d2 = logreg(x_iris2d2_train, y_iris2d2_train, np.zeros(iris2d2_train.shape[1]))
#print(w2d2)
#pred_2d2 = predict_logreg(x_iris2d2_test, w2d2)
#loss_2d2 = zero_one_loss(y_iris2d2_test, pred_2d2)
#print(loss_2d2)

##############
# Exercise 9 #
##############
mnist_digits = np.loadtxt('MNIST_179_digits.txt')
mnist_labels = np.loadtxt('MNIST_179_labels.txt')

# figure out how to initialize starting point

# initialize random staring_point
starting_point = random_start(mnist_digits)

kmeans = KMeans(n_clusters = 3, n_init = 1, init = starting_point, algorithm = 'full').fit(mnist_digits)
#print(kmeans.labels_)
#print(kmeans.labels_.shape)

p = compute_percentages(mnist_labels, kmeans.labels_)
#print(p)

centers = kmeans.cluster_centers_
c1, c2, c3 = reshape_centers(centers)
#draw_images(c1, c2, c3)

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
#print(avg_loss)

nn = KNeighborsClassifier(n_neighbors = 1)
nn.fit(xTrain, yTrain)
accK = accuracy_score(yTest, nn.predict(xTest))
#print(accK)

###############
# Exercise 10 #
###############

# the training data is the first 900 entries of mnist_digits
# use PCA on MNIST dataset
evals, evecs, mean = pca(xTrain)

# plot cumulativa variance
total = np.sum(evals)
cumulative_var = []
s = 0
for ev in evals:
    s = s + ev/total
    cumulative_var.append(s)
plt.plot(cumulative_var)
plt.xlabel('Used PCs index')
plt.ylabel('Cumulative variance')
#plt.show()
plt.close()

# run clustering with k = 3 (on projected data with 20 and 200 dimensions)
mnist_20 = mds(xTrain, 20)
mnist_200 = mds(xTrain, 200)
print(mnist_20.shape)
print(mnist_200.shape)

kmeans20 = KMeans(n_clusters = 3, n_init = 1, init = random_start(mnist_20), algorithm = 'full').fit(mnist_20)
kmeans200 = KMeans(n_clusters = 3, n_init = 1, init = random_start(mnist_200), algorithm = 'full').fit(mnist_200)

# count percentages for both dimensions
p_20 = compute_percentages(yTrain, kmeans20.labels_)
p_200 = compute_percentages(yTrain, kmeans200.labels_)
#print(p_20)
print(p_200)

# revert centroid projections back to full dimensional space for image drawing
ev20 = evecs[:, 0:20]
ev20 = np.transpose(ev20)
revert_20 = np.matmul(kmeans20.cluster_centers_, ev20)

ev200 = evecs[:, 0:200]
ev200 = np.transpose(ev200)
revert_200 = np.matmul(kmeans200.cluster_centers_, ev200)

for i in range(3):
    revert_20[i, :] += np.transpose(mean)
    revert_200[i, :] += np.transpose(mean)

#c1_20, c2_20, c3_20 = reshape_centers(revert_20)
#draw_images(c1_20.real, c2_20.real, c3_20.real)

c1_200, c2_200, c3_200 = reshape_centers(revert_200)
draw_images(c1_200.real, c2_200.real, c3_200.real)


# knn classifier and n fold validation for both dimensions
