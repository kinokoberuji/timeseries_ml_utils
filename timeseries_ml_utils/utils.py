from typing import Callable

import numpy as np


def batch_predictor_from(predictor: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    def predict_batch(batch):
        return np.stack([predictor(sample) for sample in batch], axis=0)

    return predict_batch
