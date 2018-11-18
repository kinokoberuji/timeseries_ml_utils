from unittest import TestCase
from ..denoise import regression_line_slope
import numpy as np


class TestRegression_line_slope(TestCase):

    def test_regression_line_slope(self):
        "Given"
        x = np.array([2,3,4,5,6])

        "When"
        slope = regression_line_slope(x)[0]

        "Then"
        self.assertAlmostEqual(1.0, slope, 4)
