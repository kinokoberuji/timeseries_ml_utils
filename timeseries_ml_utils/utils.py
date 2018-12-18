from typing import Callable

import numpy as np


def batch_predictor_from(predictor: Callable[[np.ndarray], np.ndarray]) -> Callable[[np.ndarray], np.ndarray]:
    def predict_batch(batch):
        return np.stack([predictor(sample) for sample in batch], axis=0)

    return predict_batch


sinusoidal_time_calculators = {
        "cos_dow": lambda df: np.cos(2 * np.pi * df.index.dayofweek / 7),
        "sin_dow": lambda df: np.sin(2 * np.pi * df.index.dayofweek / 7),
        "cos_woy": lambda df: np.cos(2 * np.pi * df.index.week / 52),
        "sin_woy": lambda df: np.sin(2 * np.pi * df.index.week / 52),
        "cos_doy": lambda df: np.cos(2 * np.pi * df.index.dayofyear / 366),
        "sin_doy": lambda df: np.sin(2 * np.pi * df.index.dayofyear / 366),
        "sin_yer": lambda df: np.sin(2 * np.pi * (df.index.year - (df.index.year // 10) * 10) / 9),
        "cos_yer": lambda df: np.cos(2 * np.pi * (df.index.year - (df.index.year // 10) * 10) / 9),
        "sin_dec": lambda df: np.sin(2 * np.pi * ((df.index.year - (df.index.year // 100) * 100) // 10) / 9),
        "cos_dec": lambda df: np.cos(2 * np.pi * ((df.index.year - (df.index.year // 100) * 100) // 10) / 9)
    }
