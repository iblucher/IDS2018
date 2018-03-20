import matplotlib.pyplot as plt
import matplotlib.cm as cm

def visualize_var(temp, c):
    cv = [0.3, 0.6, 0.9, 0.6, 0.3]
    blues = plt.get_cmap('Blues')
    for m in range(5):
        spatial_var_x = []
        spatial_var_y = []
        spatial_var = temp[m]
        for i in range(0, c - 1, 2):
            spatial_var_x.append(spatial_var[i])
            spatial_var_y.append(spatial_var[i + 1])
        plt.scatter(spatial_var_x, spatial_var_y, marker = '.', color = blues(cv[m]))
        plt.plot(spatial_var_x, spatial_var_y, color = blues(cv[m]))
        plt.plot([spatial_var_x[0], spatial_var_x[89]], [spatial_var_y[0], spatial_var_y[89]], color = blues(cv[m]))
    plt.axis('equal')
    plt.xlabel('x-coordinates')
    plt.ylabel('y-coordinates')
    plt.show()
    plt.close()
