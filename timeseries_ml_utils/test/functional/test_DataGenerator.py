from unittest import TestCase

from timeseries_ml_utils.data import *
from timeseries_ml_utils.encoders import *


class Test_DataGenerator(TestCase):

    def test_scratch(self):
        # fetch data
        data = DataFetcher(["GLD.US"], limit=100)  # FIXME use a dataframe from file
        data.fetch_data().tail()

        print(len(data.get_dataframe()))
        model_data = DataGenerator(data.get_dataframe(),
                                   {"GLD.US.Close$": identity},
                                   {"GLD.US.Close$": identity},
                                   lstm_memory_size=10, aggregation_window_size=16, batch_size=10,
                                   model_filename="/tmp/keras-foo-12.h5")

        backtest_result = model_data.back_test(lambda x: x[:, -1])
        backtest_result.plot_random_sample()
        pass
