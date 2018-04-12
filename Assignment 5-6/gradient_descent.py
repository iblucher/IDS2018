from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent():

    # initialize variables
    tol = 1.0e-10
    max_iter = 1.0e4
    x = 1
    lrate = 0.0001
    it = 0

    x_values = []
    f_values = []
    grad_values = []

    grad = 20*x - 0.5 * np.exp(-x/2)

    while np.linalg.norm(grad) > tol and it < max_iter:
        grad = 20*x - 0.5 * np.exp(-x/2)
        f = np.exp(-x/2) + 10 * (x ** 2)
        f_values.append(f)
        grad_values.append(grad)
        x_values.append(x)
        x = x - lrate * grad
        it = it + 1

    f = np.exp(-x/2) + 10 * (x ** 2)
    return f, it, x_values, f_values, grad_values

f, it, x_values, f_values, grad_values = gradient_descent()

def func(x):
    return np.exp(-x/2) + 10 * (x ** 2)

t = np.linspace(-1, 1)
for i in range(10):
    if i != 10:
        #print('oi')
        #plt.plot([x_values[i], x_values[i+1]], [f_values[i], f_values[i+1]], linewidth = 3.5, linestyle = ':', color = 'black')
    #plt.plot(t, f_values[i] + grad_values[i] * (t - x_values[i]), label = myLegend[i])

#plt.plot(t, func(t))
#plt.xlabel('Value of x')
#plt.ylabel('Value of f(x)')
#plt.legend()
#plt.show()
