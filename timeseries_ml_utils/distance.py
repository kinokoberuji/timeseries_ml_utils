from fastdtw import fastdtw

def relative_dtw(x, y):
    prediction_distance = fastdtw(x, y)[0]
    max_dist = len(y) * x.max()
    return (max_dist - prediction_distance) / max_dist
