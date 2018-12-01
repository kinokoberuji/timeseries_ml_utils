import datetime
import logging
import os
import os.path
import re
from typing import List, Dict, Callable, Tuple

import keras
import numpy as np
import pandas as pd
from keras.models import load_model
from pandas_datareader import DataReader

from timeseries_ml_utils.encoders import identity
from .callbacks import RelativeAccuracy
from .statistics import add_ewma_variance, add_sinusoidal_time, relative_dtw, r_square


class DataFetcher:
    '''Fetches data from the web into a pandas data frame and holds it in a file cache'''

    def __init__(self, symbols, data_source="stooq", limit=None, cache_path="./.cache"):
        self.file_name = cache_path + '/' + '_'.join(symbols) + "." + data_source + "." + str(limit) + '.h5'
        self.symbols  = symbols
        self.data_source = data_source
        self.limit = limit
        self.df_key = 'df'

        if not os.path.exists(cache_path):
            os.makedirs(cache_path)

        if os.path.isfile(self.file_name):
            os.remove(self.file_name)

    def fetch_data(self):
        df = pd.concat([DataReader(symbol, self.data_source).add_prefix(symbol + ".") for symbol in self.symbols],
                       axis=1,
                       join='inner') \
               .sort_index()

        df = df.dropna()

        if self.limit is not None:
            df = df.loc[df.index[-self.limit]:]

        df.to_hdf(self.file_name, key=self.df_key, mode='w')
        return df

    def get_dataframe(self):
        if not os.path.isfile(self.file_name):
            logging.info("fetching data for " + self.file_name)
            self.fetch_data()

        return pd.read_hdf(self.file_name, self.df_key)

    @classmethod
    def from_dataframes(cls, dataframes: List[pd.DataFrame]):
        # FIXME implement DataFetcher.from_dataframe
        pass


class AbstractDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self,
                 dataframe: pd.DataFrame,
                 features: List[Tuple[str, Callable]],
                 labels: List[Tuple[str, Callable]],
                 batch_size: int,
                 lstm_memory_size: int,
                 aggregation_window_size: int,
                 forecast_horizon :int,
                 training_percentage: float,
                 return_sequences: bool,
                 is_test: bool):
        'Initialization'
        logging.info("use features: " + ", ".join([f + " rescale: " + str(r) for f, r in features]))
        logging.info("use labels: " + ", ".join([l + " rescale: " + str(r) for l, r in labels]))
        self.shuffle = False
        self.dataframe = dataframe.dropna()
        self.features = features
        self.labels = labels
        self.batch_size = batch_size
        self.lstm_memory_size = lstm_memory_size
        self.aggregation_window_size = aggregation_window_size
        self.forecast_horizon = forecast_horizon
        self.training_percentage = training_percentage
        self.return_sequences = return_sequences
        self.is_test = is_test

        # calculate length
        self.min_needed_data = max(lstm_memory_size, aggregation_window_size)
        self.min_needed_data_batch = self.min_needed_data + self.batch_size
        self.min_needed_data_batch_forecast = self.min_needed_data_batch + forecast_horizon
        self.length = len(self.dataframe) - self.min_needed_data_batch_forecast

        # sanity checks
        if len(self.labels) < 1:
            raise ValueError('No labels defined')
        if len(self.features) < 1:
            raise ValueError('No features defined')
        if self.forecast_horizon < 1:
            raise ValueError('Forecast horizon needs to be >= 1')
        if self.aggregation_window_size < 1:
            raise ValueError('Aggregation window needs to be >= 1')
        if self.length < batch_size:
            raise ValueError('Not enough Data for given memory and window sizes: ' + str(self.length))

        # derived properties
        # we only know the shape (self.batch_size, self.lstm_memory_size, ??) but the number of features depend on the
        # encoder/decoder so we can not kno them a priori
        item_zero = self[0]
        self.batch_feature_shape = item_zero[0].shape
        self.batch_label_shape = item_zero[1].shape

    def __len__(self):
        'Denotes the number of batches per epoch'
        cutoff = int(self.length * self.training_percentage)
        if self.is_test:
            return self.length - cutoff
        else:
            return cutoff

    def __getitem__(self, i):
        'Generate one batch of data like [batch_size, lstm_memory_size, features]'
        features_batch, _ = self._get_features_batch(i)
        labels_batch, _ = self._get_labels_batch(i, self.labels)
        return features_batch, labels_batch

    def _get_features_batch(self, i, decoders=None):
        # offset index if test set
        features_loc = self._get_features_loc(i)
        features, index = self._build_matrix(features_loc, features_loc, decoders or self.features, True)
        return features, index

    def _get_features_loc(self, i):
        return i + int(self.length * self.training_percentage) if self.is_test else i

    def _get_end_of_features_loc(self, i):
        return self._get_features_loc(i) + self.min_needed_data

    def _get_labels_batch(self, i, decoders=None):
        # offset index if test set
        features_loc = self._get_features_loc(i)
        labels_loc = self._get_labels_loc(i)
        labels, index = self._build_matrix(labels_loc, features_loc, decoders or self.labels, self.return_sequences)
        return labels, index

    def _get_labels_loc(self, i):
        return self._get_features_loc(i) + self.forecast_horizon

    def _build_matrix(self, loc, ref_loc, column_encoders, is_lstm_aggregate):
        # aggregate windows
        # shape = ((feature/label_columns, lstm_memory_size + batch_size, window), ...)
        matrix, index = self._aggregate_window(loc, AbstractDataGenerator._get_column_names(column_encoders))

        # get reference values for encoding
        ref_values, ref_index = self._get_reference_values(ref_loc, self._get_column_names(column_encoders))

        # encode data like normalization
        matrix = self._encode(matrix, ref_values, column_encoders)

        # concatenate all feature and label vectors into one vector
        # shape = (lstm_memory_size + batch_size, window * feature/label_columns)
        matrix = self._concatenate_vectors(matrix)

        # make sliding window of lstm_memory_size
        # shape = (batchsize, lstm_memory_size, window * feature/label_columns)
        matrix = [matrix[i:i + self.lstm_memory_size] for i in range(self.batch_size)] if is_lstm_aggregate \
            else [matrix[i + self.lstm_memory_size - 1] for i in range(self.batch_size)]

        index = [index[i + self.lstm_memory_size - 1] for i in range(self.batch_size)]
        return np.array(matrix), index

    def _aggregate_window(self, loc, columns):
        # create data windows
        df = self.dataframe
        window = self.aggregation_window_size

        lstm_range = range(loc, loc + self.min_needed_data)

        # shape = (columns, lstm_memory_size + batch_size, window)
        matrix = np.array([[df[column].iloc[j:j+window].values
                            for j in lstm_range]
                           for i, column in enumerate(columns)])

        index = [df.index[j:j+window].values for j in lstm_range]
        return matrix, index

    def _get_reference_values(self, loc, columns):
        df = self.dataframe
        nr_of_rows = self.min_needed_data

        ref_index = [df.index[max(0, loc + row - 1)] for row in (range(nr_of_rows))]
        ref_values = [np.array([df[col].iloc[max(0, loc + row - 1)]
                                for row in (range(nr_of_rows))])
                      for col in columns]

        ref_values = np.stack(ref_values, axis=0)
        return ref_values, ref_index

    def _encode(self, aggregate_matrix, reference_values, encoding_functions):
        encoded = np.stack([np.array([func(aggregate_matrix[i, row], reference_values[i, row], True)
                                      for row in (range(aggregate_matrix.shape[1]))])
                            for i, (col, func) in enumerate(encoding_functions)], axis=0)

        return encoded

    def _decode_batch(self, batch, ref_values_index, encoding_functions):
        # return shape (features, batch_size, aggregation_window)
        decoded_batch = np.hstack([self._decode(item, ref_values_index[i][0], encoding_functions)
                                   for i, item in enumerate(batch)])

        decoded_indexes = [np.array(i[1]) for i in ref_values_index]

        return decoded_batch, decoded_indexes

    def _decode(self, vector, reference_values, encoding_functions):
        # first reshape a 1D vector into (nr of labels, 1, aggregation window)
        decoded = vector.reshape((len(self.labels), 1, -1))

        # now we can decode each row of each column with its associated decoder
        decoded = np.stack([np.array([func(decoded[i, row], reference_values[i, row], False)
                                      for row in (range(decoded.shape[1]))])
                            for i, (col, func) in enumerate(encoding_functions)], axis=0)

        return decoded

    def _concatenate_vectors(self, array3D):
        # shape = ((feature/label_columns, lstm_memory_size + batch_size, window), ...)
        return array3D.transpose((1, 0, 2)) \
                      .reshape((-1, self.aggregation_window_size * len(array3D)))

    def back_test(self, batch_predictor: Callable[[np.ndarray], np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        length = len(self) + self.batch_size - self.forecast_horizon

        # make a prediction
        batches = zip(*[self._back_test_batch(batch, batch_predictor) for batch in range(0, length, self.batch_size)])
        prediction = np.hstack(batches[0])
        labels = np.hstack(batches[1])
        errors = np.hstack(batches[2])
        r_squares = np.hstack(batches[3])

        stds = None  # np.apply_over_axes(np.std, errors, [1])  # expect errors.shape[0] == labels.shape[1]

        return prediction, labels, r_squares, stds

    def _back_test_batch(self, i,batch_predictor):
        # make a prediction
        prediction = self._decode_batch(self._get_labels_loc(i),
                                        batch_predictor(self._get_features_batch(i)[0]),
                                        self.labels)

        # get the labels.
        # Note that we do not want to encode the labels this time so we pass identity encoder and decoder
        identity_encoders = [(col, identity) for _, (col, _) in enumerate(self.labels)]
        labels = self._decode_batch(self._get_labels_loc(i),
                                    self._get_labels_batch(i, identity_encoders)[0],
                                    identity_encoders)

        # calculate errors between prediction and label per value
        errors = prediction - labels
        r_squares = np.array([0])

        return prediction, labels, errors, r_squares

    @staticmethod
    def _get_column_names(encoders):
        return [col for i, (col, _) in enumerate(encoders)]

    def _get_last_features(self, n=-1):
        i = self.predictive_length() - self.batch_size + n
        features_batch, index = self._get_features_batch(i)
        return features_batch[-1], index[-1]

    def predictive_length(self):
        return len(self.dataframe) - self.min_needed_data

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            raise ValueError('Shuffling not implemented yet')


class TestDataGenerator(AbstractDataGenerator):

    def __init__(self, data_generator, training_percentage: float=None):
        super(TestDataGenerator, self).__init__(data_generator.dataframe,
                                                data_generator.features,
                                                data_generator.labels,
                                                data_generator.batch_size,
                                                data_generator.lstm_memory_size,
                                                data_generator.aggregation_window_size,
                                                data_generator.forecast_horizon,
                                                training_percentage or data_generator.training_percentage,
                                                data_generator.return_sequences,
                                                True)

    def on_epoch_end(self):
        pass


class PredictiveDataGenerator(AbstractDataGenerator):

    # we need to extend DataGenerator because of windowing, encoding and decoding ...
    def __init__(self, model: keras.Model, data_generator):
        super(PredictiveDataGenerator, self).__init__(data_generator.dataframe,
                                                      data_generator.features,
                                                      data_generator.labels,
                                                      data_generator.batch_size,
                                                      data_generator.lstm_memory_size,
                                                      data_generator.aggregation_window_size,
                                                      data_generator.forecast_horizon,
                                                      1.0,
                                                      data_generator.return_sequences,
                                                      False)

        self.model = model

    def back_test(self, batch_predictor: Callable[[np.ndarray], np.ndarray] = None):
        return super(PredictiveDataGenerator, self).back_test(batch_predictor or self.model.predict)

    def predict(self, i: int):
        timedelta = self._get_time_delta()
        df = self.dataframe
        df_dates = df.index
        df_len = len(df)

        # get features and locations in dataframe
        features, index = self._get_last_features(i) if i < 0 else self._get_features_loc(i)
        start_loc_features = df.index.get_loc(index[0])
        start_loc_labels = start_loc_features + self.forecast_horizon

        # for some reason we need to fake a batch for the predict method
        features_batch = np.repeat([features], self.batch_size, axis=0)
        prediction, _ = self._decode(start_loc_labels, self.model.predict(features_batch)[-1], self.labels)
        stop_loc_labels = start_loc_labels + prediction.shape[2]

        # calculate timeline for past and predicted values
        timeline = [df_dates[i] if i < df_len else df_dates[-1] + timedelta * (i - df_len + 1)
                    for i in range(start_loc_features, stop_loc_labels)]

        # generate the prediction dataframe
        prediction_df = pd.DataFrame({
            col + " predicted": np.hstack([np.repeat(np.nan, len(timeline) - prediction.shape[-1]), prediction[i][-1]])
            for i, (col, _) in enumerate(self.labels)
        }, index=timeline)

        # join the historic dataframe
        # columns = list(set(self._get_column_names(self.features) + self._get_column_names(self.labels)))
        columns = self._get_column_names(self.labels)
        prediction_df = prediction_df.join(df[columns], how='left')

        # todo add upper and lower band
        return prediction_df

    def _get_time_delta(self):
        ix = self.dataframe.index
        time_deltas = (ix - np.roll(ix, 1)).values
        time_deltas[0] = datetime.timedelta(0)
        return time_deltas[1:].min()


class DataGenerator(AbstractDataGenerator):

    def __init__(self,
                 dataframe,  # FIXME provide a DataFetcher and use a classmethod on the DataFetcher instead
                 features: Dict[str, Callable[[np.ndarray, float, bool], np.ndarray]],
                 labels: Dict[str, Callable[[np.ndarray, float, bool], np.ndarray]],
                 batch_size: int=100, lstm_memory_size: int=52 * 5, aggregation_window_size: int=32, forecast_horizon: int=None,
                 training_percentage: float=0.8,
                 return_sequences: bool=False,
                 variances: Dict[str, float]={".*": 0.94},
                 model_filename: str="./model.h5"):
        super(DataGenerator, self).__init__(add_sinusoidal_time(add_ewma_variance(dataframe, variances)),
                                            [(col, r) for col in dataframe.columns for f, r in features.items() if re.search(f, col)],
                                            [(col, r) for col in dataframe.columns for l, r in labels.items() if re.search(l, col)],
                                            batch_size,
                                            lstm_memory_size,
                                            aggregation_window_size,
                                            aggregation_window_size if forecast_horizon is None else forecast_horizon,
                                            training_percentage,
                                            return_sequences,
                                            False)

        super(DataGenerator, self).on_epoch_end()
        self.model_filename = model_filename

    def as_test_data_generator(self, training_percentage: float=None) -> TestDataGenerator:
        return TestDataGenerator(self, training_percentage)

    def fit(self,
            model: keras.Model,
            fit_generator_args: Dict,
            relative_accuracy_function: Callable[[np.ndarray, np.ndarray], float]=relative_dtw,
            frequency: int=50,
            log_dir: str=".logs/"):

        test_data = self.as_test_data_generator()
        callback = RelativeAccuracy(test_data, relative_accuracy_function, frequency, log_dir)

        fit_generator_args["generator"] = self
        fit_generator_args["validation_data"] = test_data
        fit_generator_args["callbacks"] = [callback] + fit_generator_args.get("callbacks", [])

        hist = model.fit_generator(**fit_generator_args)

        model.save(self.model_filename)
        return hist  # TODO return PredictiveDataGenerator

    def as_predictive_data_generator(self) -> PredictiveDataGenerator:
        model = load_model(self.model_filename)
        return PredictiveDataGenerator(model, self)



