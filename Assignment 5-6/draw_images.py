import matplotlib.pyplot as plt

def draw_images(c1, c2, c3):
    # create images from cluster centers
    plt.imshow(c1)
    plt.show()
    plt.imshow(c2)
    plt.show()
    plt.imshow(c3)
    plt.show()
