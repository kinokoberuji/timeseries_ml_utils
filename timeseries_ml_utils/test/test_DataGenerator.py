import os
import unittest
from unittest import TestCase
import pandas as pd
import numpy as np
import timeseries_ml_utils.test
from ..data import DataGenerator
from timeseries_ml_utils.encoders import *

pd.options.display.max_columns = None


class TestDataGenerator(TestCase):

    def test___getitem__(self):
        pd.options.display.max_columns = None

        df = pd.DataFrame({
            "GLD.US.Close": np.arange(19.0),
            "GLD.US.Volume": np.arange(19.0)
        }, index=pd.date_range(start='01.01.2015', periods=19))

        data_generator = DataGenerator(df, {"GLD.US.Close$": identity, "GLD.US.Volume$": identity}, {"GLD.US.Close$": identity},
                                       3, 4, 5, 1, training_percentage=0.6, return_sequences=True)
        last_index = len(data_generator) - 1

        print("\n", data_generator.dataframe)

        first_batch = data_generator.__getitem__(0)

        # assert window aggregation
        self.assertEqual(data_generator._aggregate_normalized_window(last_index, [col for col, _ in data_generator.features])[-1][-1][-1], 13.)
        self.assertEqual(data_generator._aggregate_normalized_window(last_index + data_generator.forecast_horizon, [col for col, _ in data_generator.labels])[-1][-1][-1], 14.)

        # assert first batch
        self.assertEqual(data_generator.__getitem__(0)[0][0][0][0], 0.)
        self.assertEqual(data_generator.__getitem__(0)[1][0][0][0], 1.)

        # assert last batch
        last_batch = data_generator.__getitem__(last_index)
        self.assertEqual(last_batch[0][-1][-1][-1], 13.)
        self.assertEqual(last_batch[1][-1][-1][-1], 14.)

        # assert shape
        for i in range(len(data_generator)):
            item = data_generator.__getitem__(i)
            print("\n", i, item[0].shape)
            self.assertEqual(item[0].shape, (3, 4, 10))
            self.assertEqual(item[1].shape, (3, 4, 5))

        print("\n", last_batch[0])
        print("\n", last_batch[1])

        # generate test set
        test_data_generator = data_generator.as_test_data_generator()
        last_index = len(test_data_generator) - 1

        # assert last batch of test set
        self.assertEqual(test_data_generator.__getitem__(last_index)[0][-1][-1][-1], 17.)
        self.assertEqual(test_data_generator.__getitem__(last_index)[1][-1][-1][-1], 18.)

        # assert first batch of test set
        self.assertEqual(test_data_generator.__getitem__(0)[0][-1][-1][-1], 14.)
        self.assertEqual(test_data_generator.__getitem__(0)[1][-1][-1][-1], 15.)

    def test__rescale(self):
        pd.options.display.max_columns = None

        df = pd.DataFrame({
            "GLD.US.Close": np.arange(19.0) + 2.0,   # prevent division by 0 or 1
            "GLD.US.Volume": np.arange(19.0) + 2.0,  # prevent division by 0 or 1
        }, index=pd.date_range(start='01.01.2015', periods=19))

        data_generator = DataGenerator(df, {"GLD.US.Close$": normalize, "GLD.US.Volume$": identity}, {"GLD.US.Close$": normalize},
                                       3, 4, 5, 1, training_percentage=1.0, return_sequences=True)

        last_index = len(data_generator) - 1
        first_batch = data_generator.__getitem__(0)
        last_batch = data_generator.__getitem__(last_index)

        x = np.array([2,3,4,5,6])
        y = np.array([3,4,5,6,7])
        np.testing.assert_array_almost_equal(first_batch[0][0][0], np.hstack([x / 2 - 1, x]), 4)
        np.testing.assert_array_almost_equal(first_batch[1][0][0], y / 2 - 1, 4)

        x = np.array([15,16,17,18,19])
        y = np.array([16,17,18,19,20])
        np.testing.assert_array_almost_equal(last_batch[0][-1][-1], np.hstack([x / 14 - 1, x]), 4)
        np.testing.assert_array_almost_equal(last_batch[1][-1][-1], y / 15 - 1, 4)

    def test__forecast_horizon(self):
        pd.options.display.max_columns = None

        df = pd.DataFrame({
            "GLD.US.Close": np.arange(19.0) + 2.0,   # prevent division by 0 or 1
            "GLD.US.Volume": np.arange(19.0) + 2.0,  # prevent division by 0 or 1
        }, index=pd.date_range(start='01.01.2015', periods=19))

        data_generator = DataGenerator(df, {"GLD.US.Close$": normalize, "GLD.US.Volume$": identity}, {"GLD.US.Close$": normalize},
                                       2, 2, 7, 7, training_percentage=1.0, return_sequences=True)

        last_index = len(data_generator) - 1
        last_batch = data_generator.__getitem__(last_index)

        x = np.array([ 7, 8, 9,10,11,12,13])
        y = np.array([14,15,16,17,18,19,20])
        np.testing.assert_array_almost_equal(last_batch[0][-1][-1], np.hstack([x / 6 - 1, x]), 4)
        np.testing.assert_array_almost_equal(last_batch[1][-1][-1], y / 13 - 1, 4)

    def test__just_enough_data(self):
        pd.options.display.max_columns = None

        df = pd.DataFrame({
            "GLD.US.Close": np.arange(18.0) + 2.0,   # prevent division by 0 or 1
            "GLD.US.Volume": np.arange(18.0) + 2.0,  # prevent division by 0 or 1
        }, index=pd.date_range(start='01.01.2015', periods=18))

        data_generator = DataGenerator(df, {"GLD.US.Close$": identity, "GLD.US.Volume$": identity}, {"GLD.US.Close$": normalize},
                                       1, 1, 9, 9, training_percentage=1.0, return_sequences=True)

        last_index = len(data_generator) - 1
        first_batch = data_generator.__getitem__(0)
        last_batch = data_generator.__getitem__(last_index)

        np.testing.assert_array_almost_equal(first_batch[0], last_batch[0], 4)
        np.testing.assert_array_almost_equal(first_batch[1], last_batch[1], 4)

    def test__only_one_feature(self):
        pd.options.display.max_columns = None

        df = pd.DataFrame({
            "GLD.US.Volume": np.arange(18.0) + 2.0  # prevent division by 0 or 1
        }, index=pd.date_range(start='01.01.2015', periods=18))

        data_generator = DataGenerator(df, {"GLD.US.Volume$": identity}, {"GLD.US.Volume$": identity},
                                       2, 2, 7, 7, training_percentage=1.0, return_sequences=True)

        last_index = len(data_generator) - 1
        last_batch = data_generator.__getitem__(last_index)

        self.assertEqual(last_batch[0].shape, (2,2,7))
        self.assertEqual(last_batch[1].shape, (2,2,7))

    def test__not_enough_data(self):
        pd.options.display.max_columns = None

        df = pd.DataFrame({
            "GLD.US.Volume": np.arange(18.0) + 2.0  # prevent division by 0 or 1
        }, index=pd.date_range(start='01.01.2015', periods=18))

        try:
            data_generator = DataGenerator(df, {"GLD.US.Volume$": identity}, {"GLD.US.Volume$": identity},
                                           2, 2, 9, 9, training_percentage=1.0, return_sequences=True)
        except ValueError:
            self.assertTrue(True)
        else:
            self.assertTrue(False)

    def test__get_last_features(self):
        pd.options.display.max_columns = None

        df = pd.DataFrame({
            "GLD.US.Volume": np.arange(18.0) + 2.0  # prevent division by 0 or 1
        }, index=pd.date_range(start='01.01.2015', periods=18))

        data_generator = DataGenerator(df, {"GLD.US.Volume$": identity}, {"GLD.US.Volume$": identity},
                                       2, 2, 7, 7, training_percentage=1.0, return_sequences=True)

        print(data_generator.get_last_features())
        np.testing.assert_almost_equal(np.array([[12, 13, 14, 15, 16, 17, 18], [13, 14, 15, 16, 17, 18, 19]]),
                                       data_generator.get_last_features())

    def test_prediction(self):
        path = os.path.dirname(timeseries_ml_utils.test.__file__)
        df = pd.read_hdf(os.path.join(path, "resources", "gld.us.h5"), "GLD_US")

        data_generator = DataGenerator(df, {"Volume$": normalize}, {"Volume$": normalize},
                                       2, 2, 7, 7, training_percentage=1.0, return_sequences=False,  # TODO test a true case
                                       model_filename=os.path.join(path, "resources", "test-prediction-model.h5"))

        predictor = data_generator.as_predictive_data_generator()
        predicted_df = predictor.predict(-1)
        print(predicted_df)
        self.assertTrue(True)

    def test_prediction_with_label_data(self):
        path = os.path.dirname(timeseries_ml_utils.test.__file__)
        df = pd.read_hdf(os.path.join(path, "resources", "gld.us.h5"), "GLD_US")

        data_generator = DataGenerator(df, {"Volume$": normalize}, {"Volume$": normalize},
                                       2, 2, 7, 1, training_percentage=1.0, return_sequences=False,
                                       model_filename=os.path.join(path, "resources", "test-prediction-model.h5"))

        predictor = data_generator.as_predictive_data_generator()
        predicted_df = predictor.predict(-100)
        print(predicted_df)
        predicted_df.plot()
        self.assertTrue(True)
