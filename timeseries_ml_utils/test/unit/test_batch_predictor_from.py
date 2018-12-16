from unittest import TestCase
import numpy as np

from timeseries_ml_utils.utils import batch_predictor_from


class TestBatch_predictor_from(TestCase):

    def test_batch_predictor_from(self):
        """Given"""
        batch_predictor = batch_predictor_from(lambda x: x - 1)
        features_batch = np.array([
            [[1], [2], [3]],
            [[4], [5], [6]]
        ])

        """When"""
        prediction = batch_predictor(features_batch)

        """Then"""
        np.testing.assert_array_equal(np.array([[[0], [1], [2]], [[3], [4], [5]]]), prediction)
