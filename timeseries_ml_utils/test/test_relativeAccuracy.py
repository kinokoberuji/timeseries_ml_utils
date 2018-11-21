from unittest.mock import Mock

import pandas as pd

import numpy as np
from unittest import TestCase, mock

from timeseries_ml_utils.data import DataGenerator
from timeseries_ml_utils.statistics import RelativeAccuracy


class TestRelativeAccuracy(TestCase):

    def test_on_train_end(self):
        df = pd.DataFrame({
            "GLD.US.Close": np.arange(19.0),
            "GLD.US.Volume": np.arange(19.0)
        }, index=pd.date_range(start='01.01.2015', periods=19))

        data_generator = DataGenerator(df, {"GLD.US.Close$": False, "GLD.US.Volume$": False}, {"GLD.US.Close$": False},
                                       3, 4, 5, 1, training_percentage=0.6, return_sequences=False)

        model = Mock()
        model.predict = Mock(return_value=data_generator.__getitem__(1)[1] + .2)
        pd.options.display.max_columns = None

        acc = RelativeAccuracy(data_generator)
        acc.set_model(model)
        acc.on_train_end()
