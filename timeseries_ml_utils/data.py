import datetime

from pandas_datareader import DataReader
from typing import List, Dict, Callable
from keras.models import load_model
import pandas as pd
import numpy as np
import os.path
import logging
import keras
import os
import re

from .callbacks import RelativeAccuracy
from .statistics import add_ewma_variance, add_sinusoidal_time, relative_dtw


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
                 training_percentage,
                 return_sequences,
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
        self.forecast_horizon = forecast_horizon
        self.training_percentage = training_percentage
        self.return_sequences = return_sequences
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
        features_loc = self._get_features_loc(i)
        labels_loc = self._get_labels_loc(i)

        features = self._build_matrix(features_loc, self.features, True)
        labels = self._build_matrix(labels_loc, self.labels, self.return_sequences)

        return features, labels

    def _get_features_loc(self, i):
        return i + int(self.length * self.training_percentage) if self.is_test else i

    def _get_labels_loc(self, i):
        return self._get_features_loc(i) + self.forecast_horizon

    def _build_matrix(self, loc, column_encoders, is_lstm_aggregate):
        # aggregate windows
        # shape = ((feature/label_columns, lstm_memory_size + batch_size, window), ...)
        matrix = self._aggregate_normalized_window(loc, [col for col, _ in column_encoders])

        # encode data like normalization
        matrix = self._encode(loc, matrix, column_encoders)

        # concatenate all feature and label vectors into one vector
        # shape = (lstm_memory_size + batch_size, window * feature/label_columns)
        matrix = self.__concatenate_vectors(matrix)

        # make sliding window of lstm_memory_size
        # shape = (batchsize, lstm_memory_size, window * feature/label_columns)
        matrix = [matrix[i:i + self.lstm_memory_size] for i in range(self.batch_size)] if is_lstm_aggregate \
            else [matrix[i + self.lstm_memory_size - 1] for i in range(self.batch_size)]

        return np.array(matrix)

    def _aggregate_normalized_window(self, loc, columns):
        # create data windows
        df = self.dataframe
        window = self.aggregation_window_size

        lstm_range = range(loc, loc + self.lstm_memory_size + self.batch_size - 1)
        # shape = (columns, lstm_memory_size + batch_size, window)
        vector = np.array([[df[column].iloc[j:j+window].values
                            for j in lstm_range]
                           for i, column in enumerate(columns)])

        return vector

    def _encode(self, loc, vector, encoding_functions):
        df = self.dataframe

        encoded = np.array([[func(vector[i][row], df[col].iloc[max(0, loc + row - 1)], True)
                             for row in (range(vector.shape[1]))]
                            for i, (col, func) in enumerate(encoding_functions)])

        return encoded

    def _decode(self, loc, vector, encoding_functions):
        df = self.dataframe

        decoded = np.array([[func(vector[i][row], df[col].iloc[max(0, loc + row - 1)], False)
                             for row in (range(vector.shape[1]))]
                            for i, (col, func) in enumerate(encoding_functions)])

        return decoded

    def __concatenate_vectors(self, array3D):
        # shape = ((feature/label_columns, lstm_memory_size + batch_size, window), ...)
        return array3D.transpose((1, 0, 2)) \
                      .reshape((-1, self.aggregation_window_size * len(array3D)))

    def get_last_features(self):
        features_batch = self._build_matrix(self.predictive_length() - 1, self.features, True)
        return features_batch[-1]

    def predictive_length(self):
        return len(self) + self.forecast_horizon

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            raise ValueError('Shuffling not implemented yet')


class TestDataGenerator(AbstractDataGenerator):

    def __init__(self, data_generator):
        super(TestDataGenerator, self).__init__(data_generator.dataframe,
                                                data_generator.features,
                                                data_generator.labels,
                                                data_generator.batch_size,
                                                data_generator.lstm_memory_size,
                                                data_generator.aggregation_window_size,
                                                data_generator.forecast_horizon,
                                                data_generator.training_percentage,
                                                data_generator.return_sequences,
                                                True)

    def on_epoch_end(self):
        pass


class PredictiveDataGenerator(AbstractDataGenerator):

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
        # we need to extend DataGenerator because of windowing, encoding and decoding ...

    def predict(self, index: int):
        i = index if index >= 0 else self.predictive_length() + index

        if i < 0 or i >= self.predictive_length():
            raise ValueError("not enough data, available {} requested {}".format(self.predictive_length(), i))

        # get locations in dataframe
        features_loc = self._get_features_loc(i)
        labels_loc = self._get_labels_loc(i)

        # perform a prediction of one batch of size self.batch_size
        features = self._build_matrix(features_loc, self.features, True)
        prediction = self.model.predict(features)[-1]

        # prediction has shape (, window_size * features)
        # and we need to re-shape it to (label_columns, 1, window)
        # decode the prediction
        forecast = self._decode(labels_loc, prediction, self.labels)

        # get the prediction as a pandas
        columns = [col for col, _ in self.labels.items()]
        historic_df = self.dataframe[columns].iloc[features_loc:labels_loc]
        predictive_df = pd.DataFrame({"prediction": prediction}, index=self._get_prediction_dates())  # FIXME multiple labels
        df = pd.concat([historic_df, predictive_df], sort=True)

        # if i < len(self) ... then we also have labels we want to plot
        # if i < len-of-training-set  ... then we also have labels we want to plot which were part of the training
        # TODO decode eventually encoded (denoised) vectors
        # TODO scale back to original domain

        # TODO if labels present make a comparing output else use just the prediction, later we could make a back-test
        # TODO add some kind of confidence interval around the prediction
        # return a box plot with line (if labels are present)
        # https://stackoverflow.com/questions/50181989/plot-boxplot-and-line-from-pandas/50183759#50183759
        return df

    def _get_prediction_dates(self):
        df = self.dataframe
        ix = df.index

        # calculate th time deltas and fix the first delta to be zero to get an average
        time_deltas = (ix - np.roll(ix, 1)).values
        time_deltas[0] = datetime.timedelta(0)
        avg_time_delta = time_deltas.sum() / len(time_deltas)
        return [ix[-1] + avg_time_delta * i
                for i in range(self.forecast_horizon, self.aggregation_window_size + self.forecast_horizon)]


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

    def as_test_data_generator(self) -> TestDataGenerator:
        return TestDataGenerator(self)

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



