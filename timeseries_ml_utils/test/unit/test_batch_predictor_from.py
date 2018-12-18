from unittest import TestCase
import timeseries_ml_utils.test
import pandas as pd
import numpy as np
import os
from timeseries_ml_utils.utils import batch_predictor_from
from timeseries_ml_utils.utils import sinusoidal_time_calculators


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

    def test_sinusoidal_time_calculators(self):
        """Given"""
        path = os.path.dirname(timeseries_ml_utils.test.__file__)
        df = pd.read_hdf(os.path.join(path, "resources", "gld.us.h5"), "GLD_US")

        """When"""
        for sin_time, calculator in sinusoidal_time_calculators.items():
            df["trigonometric_time." + sin_time] = calculator(df)

        """Then"""
        self.assertAlmostEqual(-9.009689e-01, df["trigonometric_time.cos_dow"].iloc[-1])
        self.assertAlmostEqual(-4.338837e-01, df["trigonometric_time.sin_dow"].iloc[-1])
        self.assertAlmostEqual( 8.229839e-01, df["trigonometric_time.cos_woy"].iloc[-1])
        self.assertAlmostEqual(-5.680647e-01, df["trigonometric_time.sin_woy"].iloc[-1])
        self.assertAlmostEqual( 7.841198e-01, df["trigonometric_time.cos_doy"].iloc[-1])
        self.assertAlmostEqual(-6.206095e-01, df["trigonometric_time.sin_doy"].iloc[-1])
        self.assertAlmostEqual(-6.427876e-01, df["trigonometric_time.sin_yer"].iloc[-1])
        self.assertAlmostEqual( 7.660444e-01, df["trigonometric_time.cos_yer"].iloc[-1])
        self.assertAlmostEqual( 6.427876e-01, df["trigonometric_time.sin_dec"].iloc[-1])
        self.assertAlmostEqual( 7.660444e-01, df["trigonometric_time.cos_dec"].iloc[-1])
