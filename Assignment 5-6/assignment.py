import numpy as np
from multivarlinreg import multivarlinreg

# read in red wine dataset
wine_train = np.loadtxt('redwine_training.txt')
wine_test = np.loadtxt('redwine_testing.txt')

x_wine_train = wine_train[:, :-1]
y_wine_train = wine_train[:, -1]
x_wine_test = wine_test[:, :-1]
y_wine_test = wine_test[:, -1]

# exercise 1b (REMEMBER: remove all other features)
w_1b = multivarlinreg(x_wine_train[:, 0], y_wine_train)
print(w_1b)
