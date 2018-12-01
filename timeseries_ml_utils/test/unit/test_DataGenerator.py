from unittest import TestCase
from random import randint
from timeseries_ml_utils.data import DataGenerator
from timeseries_ml_utils.encoders import *
import timeseries_ml_utils.test
import pandas as pd
import numpy as np
import unittest
import os

pd.options.display.max_columns = None


class Test_DataGenerator(TestCase):

    def __init__(self, methodName):
        super(Test_DataGenerator, self).__init__(methodName)
        self.df = pd.DataFrame({
            "Close": np.arange(1, 21.0),
            "Volume": np.arange(1, 21.0)
        }, index=pd.date_range(start='01.01.2015', periods=20))
        self.dg = DataGenerator(
            self.df,
            {"Close$": identity},
            {"Close$": identity},
            batch_size=2,
            lstm_memory_size=3,
            aggregation_window_size=4,
            training_percentage=1.0
        )
        self.dg_test = self.dg.as_test_data_generator(0.5)

    def test_len(self):
        labels_batch, labels_index_batch = self.dg._get_labels_batch(len(self.dg)-1)
        self.assertEqual(self.df.index[-1], pd.Timestamp(labels_index_batch[-1][-1]))

    def test_predictive_length(self):
        train_len = len(self.dg)
        predict_len = self.dg.predictive_length()
        delta = self.dg.batch_size + self.dg.forecast_horizon
        self.assertEqual(train_len, predict_len - delta)

    def test_get_features_loc(self):
        loc = self.dg._get_features_loc(0)
        loc_test = self.dg_test._get_features_loc(0)
        expected_loc_test = len(self.dg) - len(self.dg_test)

        self.assertEqual(0, loc)
        self.assertEqual(expected_loc_test, loc_test)

    def test_get_end_of_features_loc(self):
        loc = self.dg._get_features_loc(0)
        eloc = self.dg._get_end_of_features_loc(0)

        expected_last_loc = len(self.df) - 1
        last_loc = self.dg._get_end_of_features_loc(self.dg.predictive_length() - 1)
        self.assertEqual(eloc - self.dg.min_needed_data, loc)
        self.assertEqual(expected_last_loc, last_loc)
        pass

    def test_get_labels_loc(self):
        loc = self.dg._get_labels_loc(0)
        self.assertEqual(self.dg.forecast_horizon, len(self.df.index[:self.dg.forecast_horizon]))
        self.assertEqual(self.dg.forecast_horizon, loc)
        self.assertEqual(self.df.index[loc], self.df.index[:self.dg.forecast_horizon + 1][-1])

    def test_aggregate_window(self):
        last_index = len(self.dg) - 1

        # labels
        expected_label = self.df["Close"][-1]
        loc_labels = self.dg._get_labels_loc(last_index)
        labels, l_index = self.dg._aggregate_window(loc_labels, [col for col, _ in self.dg.labels])
        last_label = labels[-1][-1][-1]

        # features
        expected_feature = self.df["Close"][-self.dg.forecast_horizon-1]
        loc_features = self.dg._get_features_loc(last_index)
        features, f_index = self.dg._aggregate_window(loc_features, [col for col, _ in self.dg.features])
        last_feature = features[-1][-1][-1]

        print(f_index[-1])
        print(l_index[-1])
        self.assertEqual(expected_label, last_label)
        self.assertEqual(expected_feature, last_feature)

    def test_get_reference_values(self):
        last_index = len(self.dg) - 1
        col = "Close"
        ref_val_0, ref_index_0 = self.dg._get_reference_values(0, [col])

        matrix, index = self.dg._aggregate_window(last_index, [col for col, _ in self.dg.labels])
        expected_ref_vals_max = matrix[0, :, 0] - 1
        ref_val_max, ref_index_max = self.dg._get_reference_values(last_index, [col])

        self.assertEqual(ref_val_0[0][0], self.df[col][0])
        np.testing.assert_array_equal(ref_val_max[0], expected_ref_vals_max)

    def test_encode(self):
        encoded = self.dg._encode(np.array([[[13, 14]]]), np.array([[13]]), [("", normalize)])
        np.testing.assert_array_equal(np.array([[[13, 14]]]) / 13 - 1, encoded)

    def test_concatenate_vectors(self):
        # turn shape (features, batch/lstm size, window) into (batch/lstm size, features * window)
        input = np.array([[[1, 2, 3, 4], [1, 2, 3, 4]],[[1, 2, 3, 4], [1, 2, 3, 4]]])
        expected = np.array([[1, 2, 3, 4, 1, 2, 3, 4], [1, 2, 3, 4, 1, 2, 3, 4]])
        np.testing.assert_array_equal(expected, self.dg._concatenate_vectors(input))

    def test_build_matrix(self):
        floc = self.dg._get_features_loc(0)
        lloc = self.dg._get_labels_loc(0)
        feature_matrix, feature_index = self.dg._build_matrix(floc, floc, self.dg.features, True)
        labels_matrix, labels_index = self.dg._build_matrix(lloc, lloc, self.dg.labels, False)

        self.assertEqual((self.dg.batch_size, self.dg.lstm_memory_size, self.dg.aggregation_window_size * len(self.dg.features)), feature_matrix.shape)
        self.assertEqual((self.dg.batch_size, self.dg.aggregation_window_size * len(self.dg.labels)), labels_matrix.shape)
        self.assertEqual(self.dg.forecast_horizon, self.df.index.get_loc(labels_index[0][0]) - self.df.index.get_loc(feature_index[0][0]))

    def test_get_last_features(self):
        last_features, last_index = self.dg._get_last_features(-1)
        self.assertEqual(self.df["Close"][-1], last_features[-1][-1])
        self.assertEqual(self.df.index[-1], pd.Timestamp(last_index[-1]))
        pass

    def test_decode(self):
        arr = np.array(np.repeat([1, 2, 3, 4], len(self.dg.labels)))
        decoded = self.dg._decode(arr, np.array([[1]]), [("", normalize)])
        expected_shape = (len(self.dg.labels),
                          self.dg.lstm_memory_size if self.dg.return_sequences else 1,
                          self.dg.aggregation_window_size)

        self.assertEqual(expected_shape, decoded.shape)

    def test_decode_batch(self):
        i = 0
        f_loc = self.dg._get_features_loc(i)
        fe_loc = self.dg._get_end_of_features_loc(i)

        features, features_index = self.dg._get_features_batch(i, [(col, identity) for col, _ in self.dg.features])
        ref_values_index = [self.dg._get_reference_values(fe_loc + 1 + i, self.dg._get_column_names(self.dg.labels))
                            for i in range(len(features))]

        prediction = np.vstack([np.repeat(lstm_hist[-1][-1], lstm_hist.shape[1]) for lstm_hist in features])
        decoded, decoded_index = self.dg._decode_batch(prediction, ref_values_index, self.dg.labels)

        for i in range(len(features_index)):
            self.assertEqual(pd.Timestamp(features_index[i][-1]), decoded_index[i][-2])

        pass

    def test_back_test_batch(self):
        # prediction, labels, errors, r_squares = self.dg._back_test_batch(0, lambda x: np.repeat(x[-1], len(x)))

        pass

    def backtest(self):
        # since we normalize we denormalize using (1 + x) * ref value which equals to the ref_value
        self.dg.back_test(lambda x: np.repeat(x[-1], len(x)))

        pass

    def _get_last_features(self, n=-1):
        pass

    def predictive_length(self):
        pass


class Test_DataGenerator2(Test_DataGenerator):

    def __init__(self, method_name):
        s = super(Test_DataGenerator2, self)
        s.__init__(method_name)
        s.dg = DataGenerator(
            self.df,
            {"Close$": normalize},
            {"Close$": normalize},
            batch_size=2,
            lstm_memory_size=3,
            aggregation_window_size=4,
            training_percentage=1.0
        )
        self.dg_test = self.dg.as_test_data_generator(0.5)
