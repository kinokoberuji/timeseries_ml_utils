from unittest import TestCase
from ..distance import relative_dtw
import numpy as np


class TestRelative_dtw(TestCase):
    def test_relative_dtw(self):

        x = np.array([1.2, 1.18, 1.95, 2.09])
        y = np.array([1.2, 1.18, 1.95, 2.09])

        self.assertEqual(relative_dtw(x, y), 1.0)
