def transform_labels(data):
    r = len(data)
    for i in range(r):
        if data[i] ==  0:
            data[i] = -1
    return data
