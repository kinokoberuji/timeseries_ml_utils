from scipy.stats import linregress
import numpy as np


def regression_line_slope(x: np.ndarray) -> np.ndarray:
    slope, intercept, r_value, p_value, std_err = linregress(x)
    return np.array([slope])

