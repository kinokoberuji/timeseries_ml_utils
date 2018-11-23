import os
from unittest import TestCase
from unittest.mock import Mock

import numpy as np
import pandas as pd

from ..callbacks import RelativeAccuracy
from ..data import DataGenerator


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

        acc = RelativeAccuracy(data_generator, log_dir="./.test_on_train_end")
        acc.set_model(model)
        size = -1

        try:
            acc.on_train_end()
            file = os.path.join(acc.log_dir, os.listdir(acc.log_dir)[0])
            size = os.path.getsize(file)
        finally:
            acc.clear_all_logs()

        self.assertAlmostEqual(123453, size)
