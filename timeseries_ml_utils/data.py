from pandas_datareader import DataReader
from fastdtw import fastdtw
from typing import List, Dict, Callable
import pandas as pd
import numpy as np
import os.path
import logging
import keras
import math
import sys
import os
import re


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

    def __init__(self, dataframe, features, labels,
                 batch_size, lstm_memory_size, aggregation_window_size, forecast_horizon,
                 de_noising, training_percentage,
                 return_sequences,
                 on_epoch_end_callback,
                 is_test):
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
        self.de_noising = de_noising
        self.forecast_horizon = forecast_horizon
        self.training_percentage = training_percentage
        self.return_sequences = return_sequences
        self.on_epoch_end_callback = on_epoch_end_callback
        self.is_test = is_test

        # calculate length
        self.length = len(self.dataframe) - self.batch_size - self.lstm_memory_size + 1 - self.aggregation_window_size + 1 - self.forecast_horizon + 1

        # sanity checks
        if len(self.labels) < 1:
            raise ValueError('No labels defined')
        if len(self.features) < 1:
            raise ValueError('No features defined')
        if self.forecast_horizon < 1:
            raise ValueError('Forecast horizon needs to be >= 1')
        if self.aggregation_window_size < 1:
            raise ValueError('Aggregation window needs to be >= 1')
        if self.length < 1:
            raise ValueError('Not enough Data for given memory and window sizes: ' + str(self.length))

        # derived properties
        self.batch_feature_shape = self.__getitem__(0)[0].shape
        self.batch_label_shape = self.__getitem__(0)[1].shape

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.length - int(self.length * self.training_percentage) \
            if self.is_test else int(self.length * self.training_percentage)

    def __getitem__(self, i):
        'Generate one batch of data like [batch_size, lstm_memory_size, features]'
        # offset index if test set
        index = i + int(self.length * self.training_percentage) if self.is_test else i

        # aggregate windows
        # shape = ((feature/label_columns, lstm_memory_size + batch_size, window), ...)
        features, labels = self._aggregate_normalized_window(index)

        # normalize data
        features, labels = self._normalize(index, features, labels)

        # de noise data
        # shape = ((feature/label_columns, lstm_memory_size + batch_size, window), ...)
        features, labels = self._de_noise(features, labels)

        # concatenate all feature and label vectors into one vector
        # shape = (lstm_memory_size + batch_size, window * feature/label_columns)
        features = self.__concatenate_vectors(features)
        labels = self.__concatenate_vectors(labels)

        # make sliding window of lstm_memory_size
        # shape = (batchsize, lstm_memory_size, window * feature/label_columns)
        features = [features[i:i + self.lstm_memory_size] for i in range(self.batch_size)]
        labels = [labels[i:i + self.lstm_memory_size] for i in range(self.batch_size)] if self.return_sequences \
            else [labels[i + self.lstm_memory_size - 1] for i in range(self.batch_size)]

        return np.array(features), np.array(labels)

    def _aggregate_normalized_window(self, i):
        # create data windows
        df = self.dataframe
        window = self.aggregation_window_size

        # shape = (columns, lstm_memory_size + batch_size, window)
        features_range = range(i, i + self.lstm_memory_size + self.batch_size - 1)
        features = np.array([[df[column].iloc[j:j+window].values
                              for j in features_range]
                             for i, (column, rescale) in enumerate(self.features)])

        # shape = (columns, lstm_memory_size + batch_size, window)
        labels_range = range(i + self.forecast_horizon, i + self.lstm_memory_size + self.batch_size + self.forecast_horizon - 1)
        labels = np.array([[df[column].iloc[j:j+window].values
                            for j in labels_range]
                           for i, (column, rescale) in enumerate(self.labels)])

        return features, labels

    def _normalize(self, index, features, labels):
        df = self.dataframe

        # shape = (feature / label_columns, lstm_memory_size + batch_size, window)
        normalized_features = np.array([[(features[i][row] / df[col].iloc[max(0, index + row - 1)]) -1 if rescale else features[i][row]
                                         for row in (range(features.shape[1]))]
                                        for i, (col, rescale) in enumerate(self.features)])

        normalized_labels = np.array([[(labels[i][row] / df[col].iloc[max(0, index + self.forecast_horizon + row - 1)]) -1 if rescale else labels[i][row]
                                       for row in (range(labels.shape[1]))]
                                      for i, (col, rescale) in enumerate(self.labels)])

        return normalized_features, normalized_labels

    def _de_noise(self, features, labels):
        if self.de_noising is not None:
            feature_denoiser = {i: l if re.search(r, item) else lambda x: x
                                for i, (item, rescale) in enumerate(self.features) for r, l in self.de_noising.items()}

            label_denoiser = {i: l if re.search(r, item) else lambda x: x
                              for i, (item, rescale) in enumerate(self.labels) for r, l in self.de_noising.items()}

            features = np.array([d(features[i], 'F') for i, d in feature_denoiser.items()])
            labels = np.array([d(labels[i], 'L') for i, d in label_denoiser.items()])

        return features, labels

    def __concatenate_vectors(self, array3D):
        # shape = ((feature/label_columns, lstm_memory_size + batch_size, window), ...)
        return array3D.transpose((1, 0, 2)) \
                      .reshape((-1, self.aggregation_window_size * len(array3D)))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.on_epoch_end_callback(None)

        if self.shuffle:
            raise ValueError('Shuffling not implemented yet')


class TestDataGenerator(AbstractDataGenerator):

    def __init__(self, datagenerator, model):
        super(TestDataGenerator, self).__init__(datagenerator.dataframe,
                                                datagenerator.features,
                                                datagenerator.labels,
                                                datagenerator.batch_size,
                                                datagenerator.lstm_memory_size,
                                                datagenerator.aggregation_window_size,
                                                datagenerator.forecast_horizon,
                                                datagenerator.de_noising,
                                                datagenerator.training_percentage,
                                                datagenerator.return_sequences,
                                                lambda _: None,
                                                True)
        self.model = model
        self.accuracy = pd.DataFrame({})

    def on_epoch_end(self):
        pass


class DataGenerator(AbstractDataGenerator):

    def __init__(self, dataframe,  # FIXME provide a DataFetcher and use a classmethod on the DataFetcher instead
                 features: Dict[str, bool], labels: Dict[str, bool],
                 batch_size: int=100, lstm_memory_size: int=52 * 5, aggregation_window_size: int=32, forecast_horizon: int=None,
                 training_percentage: float=0.8,
                 return_sequences: bool=False,
                 de_noising: Dict[str, Callable[[np.ndarray, str], np.ndarray]]={".*": lambda x, feature_label_flag: x},
                 variances: Dict[str, float]={".*": 0.94},
                 on_epoch_end_callback=lambda _: None):
        super(DataGenerator, self).__init__(add_sinusoidal_time(add_ewma_variance(dataframe, variances)),
                                            [(col, r) for col in dataframe.columns for f, r in features.items() if re.search(f, col)],
                                            [(col, r) for col in dataframe.columns for l, r in labels.items() if re.search(l, col)],
                                            batch_size,
                                            lstm_memory_size,
                                            aggregation_window_size,
                                            aggregation_window_size if forecast_horizon is None else forecast_horizon,
                                            de_noising,
                                            training_percentage,
                                            return_sequences,
                                            on_epoch_end_callback,
                                            False)

        super(DataGenerator, self).on_epoch_end()

    def as_test_data_generator(self, model: keras.Model=None) -> TestDataGenerator:
        return TestDataGenerator(self, model)

    def predict(self, model: keras.Model, index: int):
        # TODO get the feature as index and predict the labels
        # TODO decode eventually encoded (denoised) vectors
        # TODO scale back to original domain
        # TODO if labels present make a comparing output else use just the prediction, later we could make a back-test
        # TODO add some kind of confidence interval around the prediction
        pass


def add_ewma_variance(df: pd.DataFrame, param: float):
    for rx, l in param.items():
        for col in df.columns:
            if re.search(rx, col):
                arr = df[col].pct_change().values
                all_var = []
                var = 0

                for i in range(len(arr)):
                    v = l * var + (1 - l) * arr[i] ** 2
                    var = 0 if math.isnan(v) or math.isinf(v) else v
                    all_var.append(var)

                df[col + "_variance"] = all_var

    return df


def add_sinusoidal_time(df):
    df["trigonometric_time.cos_dow"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    df["trigonometric_time.sin_dow"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["trigonometric_time.cos_woy"] = np.cos(2 * np.pi * df.index.week / 52)
    df["trigonometric_time.sin_woy"] = np.sin(2 * np.pi * df.index.week / 52)
    df["trigonometric_time.cos_doy"] = np.cos(2 * np.pi * df.index.dayofyear / 366)
    df["trigonometric_time.sin_doy"] = np.sin(2 * np.pi * df.index.dayofyear / 366)
    df["trigonometric_time.sin_yer"] = np.sin(2 * np.pi * (df.index.year - (df.index.year // 10) * 10) / 9)
    df["trigonometric_time.cos_yer"] = np.cos(2 * np.pi * (df.index.year - (df.index.year // 10) * 10) / 9)
    df["trigonometric_time.sin_dec"] = np.sin(2 * np.pi * ((df.index.year - (df.index.year // 100) * 100) // 10) / 9)
    df["trigonometric_time.cos_dec"] = np.cos(2 * np.pi * ((df.index.year - (df.index.year // 100) * 100) // 10) / 9)
    return df
