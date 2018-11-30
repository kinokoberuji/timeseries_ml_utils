from typing import Dict
from scipy.stats import linregress
import numpy as np


class AbstractEncoderDecoder(object):
    def encode(self, x: np.ndarray, ref_value: float) -> np.ndarray:
        raise ValueError("encode function not implemented!")

    def decode(self, x: np.ndarray, ref_value: float, errors: np.ndarray) -> Dict[float, np.ndarray]:
        raise ValueError("decode function not implemented!")


class Identity(AbstractEncoderDecoder):
    def encode(self, x: np.ndarray, ref_value: float):
        return x

    def decode(self, x: np.ndarray, ref_value: float, errors: np.ndarray):
        return {0., x}


class Normalize(AbstractEncoderDecoder):
    def encode(self, x: np.ndarray, ref_value: float):
        return x / ref_value - 1

    def decode(self, x: np.ndarray, ref_value: float, errors: np.ndarray):
        return {0., (x + 1) * ref_value}


class RegressionLine:

    def __init__(self, aggregation_window_size: int):
        self.x = np.arange(0, aggregation_window_size)

    def get_encoder_decoder(self):
        return lambda x, ref, is_encode: self.encode_decode(x, ref, is_encode)

    def encode_decode(self, x, ref_value, is_encode):
        if is_encode:
            return self.encode(x, ref_value)
        else:
            return self.decode(x, ref_value)

    def encode(self, y: np.ndarray, ref_value: float):
        slope, intercept, r_value, p_value, std_err = linregress(self.x, y)
        vec = np.zeros(len(y))
        vec[0] = slope
        return vec

    def decode(self, y: np.ndarray, ref_value: float):
        # y = kx + d
        return self.x * y[0] + ref_value


def identity(x, ref_value, is_encode):
    return x


def normalize(x, ref_value, is_encode):
    if is_encode:
        return x / ref_value - 1
    else:
        return (x + 1) * ref_value

