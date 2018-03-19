import matplotlib.pyplot as plt

def visualize_var(temp, c):
    for m in range(5):
        spatial_var_x = []
        spatial_var_y = []
        spatial_var = temp[m]
        for i in range(0, c - 1, 2):
            spatial_var_x.append(spatial_var[i])
            spatial_var_y.append(spatial_var[i + 1])
        plt.scatter(spatial_var_x, spatial_var_y, marker = '.')
        plt.plot(spatial_var_x, spatial_var_y)
    plt.axis('equal')
    plt.xlabel('x-coordinates')
    plt.ylabel('y-coordinates')
    plt.show()
    plt.close()
