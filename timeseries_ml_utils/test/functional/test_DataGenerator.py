from unittest import TestCase

import timeseries_ml_utils.test
from timeseries_ml_utils.data import *
from timeseries_ml_utils.encoders import *


class Test_DataGenerator(TestCase):

    def __init__(self, methodname):
        super(Test_DataGenerator, self).__init__(methodname)
        self.path = os.path.dirname(timeseries_ml_utils.test.__file__)
        self.df = pd.read_hdf(os.path.join(self.path, "resources", "gld.us.h5"), "GLD_US")

    def test_scratch(self):
        # fetch data
        data = self.df.iloc[-60:].copy()

        model_data = DataGenerator(data,
                                   {"Close$": identity},
                                   {"Close$": identity},
                                   lstm_memory_size=10, aggregation_window_size=16, batch_size=2,
                                   model_filename="/tmp/keras-foo-12.h5")

        backtest_result = model_data.back_test(lambda x: x[:, -1])
        backtest_result.plot_random_sample()
        pass
