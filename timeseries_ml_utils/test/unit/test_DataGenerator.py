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

        if type(self) == Test_DataGenerator:
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
        labels_batch, labels_index_batch = self.dg._get_labels_batch(self.dg.get_last_index())
        self.assertEqual(self.df.index[-1], pd.Timestamp(labels_index_batch[-1][-1]))

    def test_predictive_length(self):
        train_len = len(self.dg)
        predict_len = self.dg.predictive_length()
        delta = self.dg.forecast_horizon
        self.assertEqual(train_len, predict_len - delta)

    def test_get_features_loc(self):
        self.assertEqual(len(self.dg) // 2, self.dg_test._get_features_loc(0))

    def test_get_end_of_features_loc(self):
        loc = self.dg._get_features_loc(0)
        eloc = self.dg._get_end_of_features_loc(0)

        self.assertEqual(loc + self.dg.aggregation_window_size, eloc)
        pass

    def test_get_labels_loc(self):
        loc = self.dg._get_labels_loc(0)
        self.assertEqual(self.dg.forecast_horizon, len(self.df.index[:self.dg.forecast_horizon]))
        self.assertEqual(self.dg.forecast_horizon, loc)
        self.assertEqual(self.df.index[loc], self.df.index[:self.dg.forecast_horizon + 1][-1])

    def test_aggregate_window(self):
        last_index = self.dg.get_last_index()

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
        last_index = self.dg.get_last_index()
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

    def test_get_labels_batch(self):
        last_index = self.dg.get_last_index()
        features_ref_encoders = [(col, lambda x, ref, _: np.repeat(x[-1], len(x))) for col, _ in self.dg.labels]
        decoders = [(col, lambda x, ref, _: np.repeat(ref, len(x))) for col, _ in self.dg.labels]

        features_batch, _ = self.dg._get_features_batch(last_index)
        features_ref_batch, _ = self.dg._get_features_batch(last_index, features_ref_encoders)
        np.testing.assert_array_equal(features_batch[:, :, -1], features_ref_batch[:, :, 0])

        labels_ref_batch, labels_index = self.dg._get_labels_batch(last_index, decoders)

        self.assertEqual(self.df.index[-1], pd.Timestamp(labels_index[-1][-1]))
        self.assertEqual(features_ref_batch[-1, -1, 0], labels_ref_batch[-1, -1])

    def test_decode(self):
        arr = np.array(np.repeat([1, 2, 3, 4], len(self.dg.labels)))
        decoded = self.dg._decode(arr, np.array([[1]]), [("", normalize)])
        expected_shape = (len(self.dg.labels),
                          self.dg.lstm_memory_size if self.dg.return_sequences else 1,
                          self.dg.aggregation_window_size)

        self.assertEqual(expected_shape, decoded.shape)

    def test_get_decode_ref_values(self):
        last_index = self.dg.get_last_index()

        decoders = [(col, lambda x, ref, _: np.repeat(ref, len(x))) for col, _ in self.dg.labels]
        labels_batch, labels_index = self.dg._get_labels_batch(last_index, decoders)

        ref_values, ref_index = self.dg._get_decode_ref_values(last_index, self.dg._get_column_names(self.dg.labels))
        np.testing.assert_array_equal(labels_batch[:, 0], ref_values[:, 0, 0])

    def test_decode_batch(self):
        last_index = len(self.dg) - 1

        # shape (batch_size, features, lstm_hist, aggregation)
        features, feature_index = self.dg._get_features_batch(last_index)
        decoded_batch, index, ref_index = self.dg._decode_batch(last_index, lambda x: x[:, -1], self.dg.labels)
        np.testing.assert_array_equal(features[-1, -1], decoded_batch[-1, -1, -1])

    def test_back_test_batch(self):
        prediction, labels, errors, r_squares = self.dg._back_test_batch(0, lambda x: x[:, -1])
        expected = labels[-1] - self.dg.forecast_horizon
        np.testing.assert_array_equal(expected, prediction[-1, -1, -1])

    def test_backtest(self):
        column_decoders = [(col, lambda x, ref, _: np.repeat(ref, len(x))) for col, _ in self.dg.labels]
        first_batch_features, first_batch_labels = self.dg[0]
        last_batch_features, last_batch_labels = self.dg[self.dg.get_last_index()]

        # the last value of the features needs to equal the ref value of the decoder
        prediction_ref, labels_ref, _, _ = self.dg.back_test(lambda x: x[:, -1], column_decoders)
        self.assertEqual(first_batch_features[0, -1, -1], prediction_ref[0, 0, 0, -1])
        self.assertEqual(last_batch_features[-1, -1, -1], prediction_ref[0, -1, 0, -1])

        # the prediction is just the last window of the feattures
        prediction, labels, r_squares, stds = self.dg.back_test(lambda x: x[:, -1])
        np.testing.assert_array_equal(first_batch_features[0, -1], prediction[0, 0, 0])
        np.testing.assert_array_equal(last_batch_features[-1, -1], prediction[-1, -1, -1])

        np.testing.assert_array_equal(last_batch_labels[-1], labels[-1, -1])


class Test_DataGenerator_aggregation1(Test_DataGenerator):

    def __init__(self, method_name):
        super(Test_DataGenerator_aggregation1, self).__init__(method_name)
        self.dg = DataGenerator(
            self.df,
            {"Close$": identity},
            {"Close$": identity},
            batch_size=1,
            lstm_memory_size=1,
            aggregation_window_size=1,
            training_percentage=1.0
        )
        self.dg_test = self.dg.as_test_data_generator(0.5)

    def test_len(self):
        length = len(self.dg)
        length_test = len(self.dg_test)

        self.assertEqual(len(self.df) - 1, length)
        self.assertEqual(len(self.df) // 2 - 1, length_test)
        super(Test_DataGenerator_aggregation1, self).test_len()

    def test_concatenate_vectors(self):
        pass

    def test_decode(self):
        pass


# class Test_DataGenerator_swapped_lstm_batch_size(Test_DataGenerator):
#
#     def __init__(self, method_name):
#         super(Test_DataGenerator_swapped_lstm_batch_size, self).__init__(method_name)
#         self.dg = DataGenerator(
#             self.df,
#             {"Close$": identity},
#             {"Close$": identity},
#             batch_size=2,
#             lstm_memory_size=4,
#             aggregation_window_size=3,
#             training_percentage=1.0
#         )
#         self.dg_test = self.dg.as_test_data_generator(0.5)
#
#     def test_switched_shape(self):
#         self.assertEqual((self.dg.batch_size, self.dg.lstm_memory_size, self.dg.aggregation_window_size),
#                          self.dg.batch_feature_shape)


# TODO allow forecast wirndow 0, try to leaarn linear regression
# TODO make a test set for return_sequence = True
# TODO test multiple features
