from unittest import TestCase
import numpy as np

from timeseries_ml_utils.encoders import *


class TestNormalize(TestCase):

    def test_normalize(self):
        x = np.array([1, 2, 3, 4])
        y = 5

        encoded = normalize(x, y, True)
        decoded = normalize(encoded, y, False)
        np.testing.assert_array_almost_equal(x, decoded)

    def test_log_returns(self):
        x = np.array([1, 2, 3, 4])
        y = 5

        encoded = log_returns(x, y, True)
        decoded = log_returns(encoded, y, False)
        np.testing.assert_array_almost_equal(x, decoded)
