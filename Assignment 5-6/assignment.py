import numpy as np
from multivarlinreg import multivarlinreg
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
print(w_2b)

# exercise 2c
w_2c = multivarlinreg(x_wine_train, y_wine_train)
print(w_2c)

# exercise 3
rm_3b = rmse(t)
