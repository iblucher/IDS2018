import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, zero_one_loss
from sklearn.ensemble import RandomForestClassifier
from multivarlinreg import multivarlinreg
from predict_linreg import predict_linreg
from rmse import rmse
from gradient_descent import gradient_descent
from plot_iris import plot_iris
from transform_labels import transform_labels
from logreg import logreg
from predict_logreg import predict_logreg

# read in red wine dataset
wine_train = np.loadtxt('redwine_training.txt')
wine_test = np.loadtxt('redwine_testing.txt')

x_wine_train = wine_train[:, :-1]
y_wine_train = wine_train[:, -1]
x_wine_test = wine_test[:, :-1]
y_wine_test = wine_test[:, -1]

# exercise 2b
w_2b = multivarlinreg(x_wine_train[:, 0], y_wine_train)
#print(w_2b)

# exercise 2c
w_2c = multivarlinreg(x_wine_train, y_wine_train)
#print(w_2c)

# exercise 3
t_3b = predict_linreg(x_wine_train[:, 0], w_2b)
#print(t_3b)
rm_3b = rmse(y_wine_test, t_3b)
#print(rm_3b)

t_3c = predict_linreg(x_wine_train, w_2c)
rm_3c = rmse(y_wine_test, t_3c)
#print(rm_3c)

# exercise 4 and 5 (compare with KNN classifier from Assignment 3)

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

# exercise 6
#gradient_descent()

# exercise 7
iris2d1_train = np.loadtxt('Iris2D1_train.txt')
iris2d1_test = np.loadtxt('Iris2D1_test.txt')
iris2d2_train = np.loadtxt('Iris2D2_train.txt')
iris2d2_test = np.loadtxt('Iris2D2_test.txt')
print(iris2d2_test[:, -1])

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

print(y_iris2d2_test)

x_iris2d1_train = iris2d1_train[:, :-1]
x_iris2d1_test = iris2d1_test[:, :-1]
x_iris2d2_train = iris2d2_train[:, :-1]
x_iris2d2_test = iris2d2_test[:, :-1]

# insert column of 1s into x matrices
x_iris2d1_train = np.c_[np.ones(x_iris2d1_train.shape[0]), x_iris2d1_train]
x_iris2d1_test = np.c_[np.ones(x_iris2d1_test.shape[0]), x_iris2d1_test]
x_iris2d2_train = np.c_[np.ones(x_iris2d2_train.shape[0]), x_iris2d2_train]
x_iris2d2_test = np.c_[np.ones(x_iris2d2_test.shape[0]), x_iris2d2_test]

ir, ic = iris2d1_train.shape
#w2d1 = logreg(x_iris2d1_train, y_iris2d1_train, np.zeros(ic))
#pred_2d1 = predict_logreg(x_iris2d1_test, w2d1)
#loss_2d1 = zero_one_loss(y_iris2d1_test, pred_2d1)
#print(loss_2d1)

w2d2 = logreg(x_iris2d2_train, y_iris2d2_train, np.zeros(ic))
pred_2d2 = predict_logreg(x_iris2d2_test, w2d2)
print(y_iris2d2_test)
print(pred_2d2)
loss_2d2 = zero_one_loss(y_iris2d2_test, pred_2d2)
print(loss_2d2)
