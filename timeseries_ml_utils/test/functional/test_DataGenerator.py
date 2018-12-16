import uuid
from unittest import TestCase
import numpy as np
from keras import Sequential
from keras.layers import LSTM
from numpy.random.mtrand import RandomState

import timeseries_ml_utils.test
from timeseries_ml_utils.data import *
from timeseries_ml_utils.encoders import *
from timeseries_ml_utils.utils import batch_predictor_from


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

    def test_log_returns(self):
        data = self.df.iloc[-60:].copy()
        sigma = 0.0060837
        rand = RandomState(12)

        def log_return_encoder(y, ref, is_encoding):
            if is_encoding:
                return np.log(y / ref)
            else:
                return np.exp(y) * ref

        def batch_log_return_predictor(batch):
            return np.array([np.array([[rand.lognormal(0, sigma) - 1.0]]) for _ in batch])

        model_log_return = DataGenerator(data,
                                         {"Close$": log_return_encoder},
                                         {"Close$": log_return_encoder},
                                         lstm_memory_size=1, aggregation_window_size=1, batch_size=1,
                                         model_filename="/tmp/log_normal_1.h5")

        backtest_log_return = model_log_return.back_test(batch_log_return_predictor)

        # check confusion matrix
        confusion_matrix = backtest_log_return.confusion_matrix()["Close"]
        print("Phi: {}".format(confusion_matrix.MCC))
        self.assertEqual(8, confusion_matrix.TP)
        self.assertEqual(13, confusion_matrix.TN)
        self.assertEqual(12, confusion_matrix.FP)
        self.assertEqual(15, confusion_matrix.FN)

        # we expect some r2 which is maximum as good as just using the last value (reference value)
        r2 = backtest_log_return.hist()["Close"]
        print("best r2: {}".format(r2[1].max()))
        self.assertTrue(r2[1].max() <= .8)

    def test_slope_estimation(self):
        data = self.df.iloc[-60:].copy()

        def slope_encoder(y, ref, encode):
            if encode:
                x = np.arange(0, len(y))
                slope, intercept, r_value, p_value, std_err = linregress(x, y)
                vec = np.zeros(len(y))
                vec[0] = slope
                return vec
            else:
                slope = y[0]
                return np.array([ref + slope * i for i in range(len(y))])

        def batch_predictor(batch):
            return np.array([sample[-1] for sample in batch])

        model_data = DataGenerator(data,
                                   {"Close$": slope_encoder},
                                   {"Close$": slope_encoder},
                                   lstm_memory_size=1, aggregation_window_size=16, batch_size=10,
                                   model_filename="/tmp/slope_1.h5")

        backtest = model_data.back_test(batch_predictor)

        confusion_matrix = backtest.confusion_matrix()["Close"]
        print("Phi: {}".format(confusion_matrix.MCC))
        # we would expect 16x16 but 3 cases never happened
        self.assertEqual((13, 13), confusion_matrix.toarray().shape)

        hist = backtest.hist()["Close"]
        print("expected r2: {}".format(hist[1][np.argmax(hist[0])]))
        self.assertTrue(hist[1].max() <= .2)

    # this is rather an experiment as a test but this should run without any exception
    def test_slope_slope_estimation(self):
        data = self.df.iloc[-80:].copy()

        def slope_encoder(y, ref, encode):
            if encode:
                x = np.arange(0, len(y))
                slope, intercept, r_value, p_value, std_err = linregress(x, y)
                vec = np.zeros(len(y))
                vec[0] = slope
                return vec
            else:
                slope = y[0]
                return np.array([ref + slope * i for i in range(len(y))])

        # get the slope of the slopes, predict the next slope
        def predictor(sample):
            x = np.arange(0, len(sample))
            y = sample[:, 0]
            slope, intercept, r_value, p_value, std_err = linregress(x, y)
            slope_hat = intercept + len(sample) * intercept
            return np.repeat(slope_hat, sample.shape[1])

        # as we need to predict in batches
        batch_predictor = batch_predictor_from(predictor)

        model_data = DataGenerator(data,
                                   {"Close$": slope_encoder},
                                   {"Close$": slope_encoder},
                                   lstm_memory_size=32, aggregation_window_size=16, batch_size=2,
                                   model_filename="/tmp/slope_2.h5")

        backtest = model_data.back_test(batch_predictor)

        confusion_matrix = backtest.confusion_matrix()["Close"]
        print("Phi: {}".format(confusion_matrix.MCC))
        # we would expect 16x16 but 6 cases never happened
        self.assertEqual((10, 10), confusion_matrix.toarray().shape)

        hist = backtest.hist()["Close"]
        print("expected r2: {}".format(hist[1][np.argmax(hist[0])]))

    def test_keras_model(self):
        # encoders and decoders
        linreg = RegressionLine(16)

        # fetch data
        data = DataFetcher(["GLD.US"], limit=350)
        print(data.fetch_data().tail())
        print(len(data.get_dataframe()))

        model_data = DataGenerator(data.get_dataframe(),
                                   {"^trigonometric": identity,
                                    "(Open|High|Low|Close)$": linreg.encode_decode},
                                   {"GLD.US.Close$": linreg.encode_decode},
                                   aggregation_window_size=16, batch_size=10)

        print(model_data.batch_feature_shape)
        print(model_data.batch_label_shape)

        model = Sequential(name="LSTM-Model-1")
        model.add(LSTM(model_data.batch_label_shape[-1],
                       name="LSTM-Layer-1",
                       batch_input_shape=model_data.batch_feature_shape,
                       activation='tanh',
                       dropout=0,
                       recurrent_dropout=0,
                       stateful=True,
                       return_sequences=model_data.return_sequences))

        model.compile("Adam", loss="mse", metrics=['mae', 'acc'])

        train_args = {"epochs": 1,
                      "use_multiprocessing": True,
                      "workers": 4,
                      "shuffle": False,
                      "verbose": 1}

        fit = model_data.fit(model, train_args, frequency=10, relative_accuracy_function=r_square)
        self.assertTrue(True)

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
