import matplotlib.pyplot as plt

def plot_iris(iris, flag):
    r, c = iris.shape
    for i in range(r):
        if iris[i, 2] == 1:
            plt.scatter(iris[i, 0], iris[i, 1], c = 'b')
        elif iris[i, 2] == 0:
            plt.scatter(iris[i, 0], iris[i, 1], c = 'r')
    plt.xlabel('Feature 0')
    if flag == 1:
        plt.ylabel('Feature 2')
    elif flag == 2:
        plt.ylabel('Feature 1')
    plt.axis('equal')
    plt.show()
