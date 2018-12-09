import uuid
from unittest import TestCase
import numpy as np
import timeseries_ml_utils.test
from timeseries_ml_utils.data import *
from timeseries_ml_utils.encoders import *


class Test_DataGenerator(TestCase):

    def __init__(self, methodname):
        super(Test_DataGenerator, self).__init__(methodname)
        self.path = os.path.dirname(timeseries_ml_utils.test.__file__)
        self.df = pd.read_hdf(os.path.join(self.path, "resources", "gld.us.h5"), "GLD_US")

    def test_get_measures(self):
        data = self.df.iloc[-60:].copy()

        model_data = DataGenerator(data,
                                   {"Close$": identity},
                                   {"Close$": identity},
                                   lstm_memory_size=10, aggregation_window_size=16, batch_size=2,
                                   model_filename="/tmp/{}.h5".format(str(uuid.uuid4())))

        backtest_result = model_data.back_test(lambda x: x[:, -1])
        predictions, labels, r_squares, standard_deviations = backtest_result.get_measures()
        self.assertEqual(predictions.shape, labels.shape)
        np.testing.assert_almost_equal(np.array([-42.781, -34.638, -26.494, -18.351, -10.208, -2.065]),
                                       np.histogram(r_squares[0, :, 0], bins=5)[1], decimal=3)

    def test_scratch(self):
        data = self.df.iloc[-60:].copy()

        model_data = DataGenerator(data,
                                   {"Close$": identity},
                                   {"Close$": identity},
                                   lstm_memory_size=10, aggregation_window_size=16, batch_size=2,
                                   model_filename="/tmp/{}.h5".format(str(uuid.uuid4())))

        backtest_result = model_data.back_test(lambda x: x[:, -1])
        print(backtest_result.confusion_matrix())
        backtest_result.plot_random_sample()
