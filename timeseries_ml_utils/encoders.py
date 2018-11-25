from typing import Dict

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


def identity(x, ref_value, is_encode):
    return x


def normalize(x, ref_value, is_encode):
    if is_encode:
        return x / ref_value - 1
    else:
        return (x + 1) * ref_value

