def transform_labels(data):
    r, c = data.shape
    for i in range(r):
        if data[i, 2] ==  0:
            data[i, 2] = -1
    return data
