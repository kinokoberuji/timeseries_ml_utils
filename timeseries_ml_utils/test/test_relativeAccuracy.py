import os
import unittest
from unittest import TestCase
from unittest.mock import Mock
from timeseries_ml_utils.encoders import *
import numpy as np
import pandas as pd

from ..callbacks import RelativeAccuracy
from ..data import DataGenerator


# FIXME test re-implemented accuracy callback and move test to unit/functional
@unittest.skip("need to be re-implemented s accuracy measure needs re-thinking")
class TestRelativeAccuracy(TestCase):

    def test_relative_accuracy(self):
        acc = TestRelativeAccuracy.get_RelativeAccuracy()
        r2, features, labels, predictions, errors = acc._relative_accuracy()

        self.assertTrue(features.shape == (6,4,10))
        self.assertTrue(predictions.shape == (6, 5))
        self.assertTrue(labels.shape == (6, 5))
        self.assertTrue(r2.shape == (6,))
        self.assertTrue(errors.shape == (5,))

    def test_on_train_end(self):

        acc = TestRelativeAccuracy.get_RelativeAccuracy()
        acc.on_train_begin(None)
        size = -1

        try:
            acc.on_train_end()
            file = os.path.join(acc.log_dir, os.listdir(acc.log_dir)[0])
            size = os.path.getsize(file)
        finally:
            acc.clear_all_logs()

        self.assertTrue(size > 10000)

    @staticmethod
    def get_RelativeAccuracy():
        df = pd.DataFrame({
            "GLD.US.Close": np.arange(19.0),
            "GLD.US.Volume": np.arange(19.0)
        }, index=pd.date_range(start='01.01.2015', periods=19))

        data_generator = DataGenerator(df, {"GLD.US.Close$": identity, "GLD.US.Volume$": identity}, {"GLD.US.Close$": identity},
                                       3, 4, 5, 1, training_percentage=0.6, return_sequences=False)

        model = Mock()
        model.predict = Mock(return_value=data_generator.__getitem__(1)[1] + .2)
        pd.options.display.max_columns = None

        acc = RelativeAccuracy(data_generator, log_dir="./.test_on_train_end")
        acc.set_model(model)
        return acc
