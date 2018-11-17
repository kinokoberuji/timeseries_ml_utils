from fastdtw import fastdtw
import numpy as np


def relative_dtw(x: np.ndarray, y: np.ndarray) -> float:
    prediction_distance = fastdtw(x, y)[0]
    max_dist = len(y) * x.max()
    return (max_dist - prediction_distance) / max_dist
