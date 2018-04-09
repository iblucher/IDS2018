import numpy as np
from multivarlinreg import multivarlinreg
from predict_linreg import predict_linreg
from rmse import rmse

# read in red wine dataset
wine_train = np.loadtxt('redwine_training.txt')
wine_test = np.loadtxt('redwine_testing.txt')

x_wine_train = wine_train[:, :-1]
y_wine_train = wine_train[:, -1]
x_wine_test = wine_test[:, :-1]
y_wine_test = wine_test[:, -1]

# exercise 2b (REMEMBER: remove all other features)
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
print(rm_3c)
