import datetime
import logging
import math
import io
import os
import os.path
import re
import tempfile
import uuid
import keras
import numpy as np
import pandas as pd
import cloudpickle
from typing import List, Dict, Callable, Tuple
from keras.models import load_model
from pandas_datareader import DataReader
from timeseries_ml_utils.encoders import identity
from timeseries_ml_utils.utils import sinusoidal_time_calculators
from .callbacks import *
from .statistics import *


class DataFetcher:
    """Fetches data from the web into a pandas data frame and holds it in a file cache"""

    def __init__(self, symbols, data_source="stooq", limit=None, cache_path="./.cache"):
        self.file_name = cache_path + '/' + '_'.join(symbols) + "." + data_source + "." + str(limit) + '.h5'
        self.symbols = symbols
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

    def get_dataframe(self) -> pd.DataFrame:
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
                 variances: List[Tuple[str, float]],
                 is_test: bool):
        'Initialization'
        logging.info("use features: " + ", ".join([f + " rescale: " + str(r) for f, r in features]))
        logging.info("use labels: " + ", ".join([l + " rescale: " + str(r) for l, r in labels]))
        self.shuffle = False
        self.dataframe = self._add_sinusoidal_time(self._add_ewma_variance(dataframe, variances)).dropna()
        self.features = features
        self.labels = labels
        self.batch_size = batch_size
        self.lstm_memory_size = lstm_memory_size
        self.aggregation_window_size = aggregation_window_size
        self.forecast_horizon = forecast_horizon
        self.training_percentage = training_percentage
        self.return_sequences = return_sequences
        self.variances = variances
        self.is_test = is_test

        # calculate length
        self.min_needed_data = lstm_memory_size + aggregation_window_size - 1
        self.min_needed_aggregation_data = self.lstm_memory_size + self.batch_size - 1
        # we need at least one full batch so we need to add the batch size
        self.min_needed_data_batch = self.min_needed_data + (self.batch_size - 1)
        self.min_needed_data_batch_forecast = self.min_needed_data_batch + forecast_horizon
        self.length = len(self.dataframe) - self.min_needed_data_batch_forecast + 1

        # sanity checks
        if len(self.labels) < 1:
            raise ValueError('No labels defined')
        if len(self.features) < 1:
            raise ValueError('No features defined')
        if self.forecast_horizon < 0:
            raise ValueError('Forecast horizon needs to be >= 0')
        if self.aggregation_window_size < 1:
            raise ValueError('Aggregation window needs to be >= 1')
        if self.length < batch_size:
            raise ValueError('Not enough Data for given batch size of memory and window sizes: ' + str(self.length))

        # derived properties
        # label identity decoders / encoders used for back testing of time-series
        self.label_identity_encoders = [(col, identity) for _, (col, _) in enumerate(self.labels)]

        # we only know the shape (self.batch_size, self.lstm_memory_size, ??) but the number of features depend on the
        # encoder/decoder so we can not know them in advance
        item_zero = self[0]
        self.batch_feature_shape = item_zero[0].shape
        self.batch_label_shape = item_zero[1].shape

    @staticmethod
    def _add_ewma_variance(df, param):
        for col, l in param:
            arr = df[col].pct_change().values
            all_var = []
            var = 0

            for i in range(len(arr)):
                v = l * var + (1 - l) * arr[i] ** 2
                var = 0 if math.isnan(v) or math.isinf(v) else v
                all_var.append(var)

            df[col + "_variance"] = all_var

        return df

    @staticmethod
    def _add_sinusoidal_time(df):
        for sin_time, calculator in sinusoidal_time_calculators.items():
            df["trigonometric_time." + sin_time] = calculator(df)

        return df

    def get_df_columns(self):
        return self.dataframe.columns.values.tolist()

    def get_last_index(self):
        return len(self) - 1

    def get_all_batches(self):
        return [sample for i in range(len(self)) for sample in self[i]]

    def __len__(self):
        'Denotes the number of batches per epoch'
        cutoff = math.floor(self.length * self.training_percentage)
        if self.is_test:
            return self.length - cutoff
        else:
            return cutoff

    def __getitem__(self, i):
        'Generate one batch of data like [batch_size, lstm_memory_size, features]'
        features_batch, _ = self._get_features_batch(i)
        labels_batch, _ = self._get_labels_batch(i, self.labels)
        return features_batch, labels_batch

    def _get_features_batch(self, i, encoders=None):
        # offset index if test set
        features_loc = self._get_features_loc(i)
        features, index = self._build_matrix(features_loc, features_loc, encoders or self.features, True)
        return features, index

    def _get_features_loc(self, i):
        return i + int(self.length * self.training_percentage) if self.is_test else i

    def _get_end_of_features_loc(self, i):
        return self._get_features_loc(i) + self.aggregation_window_size

    def _get_labels_batch(self, i, encoders=None):
        # offset index if test set
        labels_loc = self._get_labels_loc(i)
        ref_loc = self._get_end_of_features_loc(i)
        labels, index = self._build_matrix(labels_loc, ref_loc, encoders or self.labels, self.return_sequences)
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

        # make sliding window of lstm_memory_size
        # shape = (batchsize, lstm_memory_size, window * feature/label_columns)
        matrix = [matrix[i:i + self.lstm_memory_size] for i in range(self.batch_size)] if is_lstm_aggregate \
            else [matrix[i + self.lstm_memory_size - 1] for i in range(self.batch_size)]

        # get data frames indices of matrix
        index = [index[i + self.lstm_memory_size - 1] for i in range(self.batch_size)]

        return np.array(matrix), index

    def _aggregate_window(self, loc, columns):
        # create data windows
        df = self.dataframe
        window = self.aggregation_window_size

        lstm_range = range(loc, loc + self.min_needed_aggregation_data)

        # shape = (columns, lstm_memory_size + batch_size, window)
        matrix = np.array([[df[column].iloc[j:j+window].values
                            for j in lstm_range]
                           for i, column in enumerate(columns)])

        index = [df.index[j:j+window].values for j in lstm_range]
        return matrix, index

    def _get_reference_values(self, loc, columns):
        df = self.dataframe
        nr_of_rows = self.min_needed_aggregation_data

        ref_index = [df.index[max(0, loc + row - 1)] for row in (range(nr_of_rows))]
        ref_values = [np.array([df[col].iloc[max(0, loc + row - 1)]
                                for row in (range(nr_of_rows))])
                      for col in columns]

        ref_values = np.stack(ref_values, axis=0)
        return ref_values, ref_index

    def _encode(self, aggregate_matrix, reference_values, encoding_functions):

        encoded = np.array([np.hstack([func(aggregate_matrix[i, row], reference_values[i, row], True)
                                       for i, (col, func) in enumerate(encoding_functions)])
                            for row in (range(aggregate_matrix.shape[1]))])

        return encoded

    def _predict(self, batch_predictor: Callable[[np.ndarray], np.ndarray], i: int = -1):
        assert i < 0

        i = self.predictive_length() + i
        decoded_batch, index, batch_ref_values, batch_ref_index = self._decode_batch(i, batch_predictor, self.labels)
        return decoded_batch[:, -1], index[-1], batch_ref_values[-1], batch_ref_index[-1]

    def back_test(self,
                  batch_predictor: Callable[[np.ndarray], np.ndarray],
                  column_decoders: List[Tuple[str, Callable[[np.ndarray, float], np.ndarray]]] = None,
                  quality_measure: Callable[[np.ndarray, np.ndarray, np.ndarray], float] = get_r_squared,
                  predict_labels: bool = True)-> BackTestHistory:
        length = len(self)

        # make a prediction for avery batch and zip all batches
        batches = list(zip(*[self._back_test_batch(batch, batch_predictor, column_decoders, predict_labels)
                             for batch in range(length)]))

        # disassemble the zipped batches to its column components
        # shape: features, batch-size, 1, aggregation
        prediction = np.hstack(batches[0])
        labels = np.hstack(batches[1])
        errors = np.hstack(batches[2])
        reference_values = np.hstack(batches[3])
        ref_index = [y for x in batches[4] for y in x]

        # measure the quality of the prediction, this should allow a comparison between models
        try:
            prediction_quality = np.reshape([quality_measure(labels[i], prediction[i], reference_values)
                                             for i in np.ndindex(prediction.shape[:-1])], prediction.shape[:-1])
        except ValueError as ve:
            if "cannot reshape" in str(ve):
                ve.message = ve.message + " are you sure about predicting labels (as time-series)?"
            raise

        # # calculate an r2 for each batch and each lstm output sequence
        # if errors.shape[-1] == 1:
        #     # the best prediction we can do is use the reference value and we compare how much better we are
        #     # for an r2 of 1 we want to be at least as close one percent of the error to the last value (ref value)
        #     # r_squares = (1 - (errors / (labels - ref_values) ** 2))[:, :, -1]
        #     r_squares = (1 - (errors / ((labels - ref_values) ** 2 * 0.01)))[:, :, -1]
        # else:
        #     r_squares = np.reshape([r2_score(labels[i], prediction[i]) for i in np.ndindex(prediction.shape[:-1])],
        #                            prediction.shape[:-1])

        standard_deviations = np.array([errors[i, :, -1, j].std()
                                        for i in range(errors.shape[0]) for j in range(errors.shape[-1])])

        return BackTestHistory(self._get_column_names(self.labels), prediction, reference_values, ref_index, labels,
                               prediction_quality, standard_deviations)

    def _back_test_batch(self, i, batch_predictor, decoders=None, predict_labels=True):
        # make a prediction
        prediction_enc_dec = decoders or self.labels
        prediction, _, _, batch_ref_index = self._decode_batch(i, batch_predictor, prediction_enc_dec)

        # get the labels.
        # Note that we do not want to encode the labels if we try to predict them (so we pass identity en-/decoder)
        # if we try to predict some encoding of labels i.e. classes we apply the same en-/decoder as for the prediction
        label_enc_dec = self.label_identity_encoders if predict_labels else prediction_enc_dec
        labels, _, ref_values, _ = self._decode_batch(i,
                                                      lambda _: self._get_labels_batch(i, label_enc_dec)[0],
                                                      label_enc_dec)

        # calculate errors between prediction and label per value
        # this should even be correct for one hot encoded vectors
        errors = (prediction - labels) ** 2

        # reshape the reference values
        ref_values = ref_values.reshape(prediction.shape[:-1])
        return prediction, labels, errors, ref_values, batch_ref_index

    def _decode_batch(self, i, predictor_function, decoding_functions):
        # get values
        features, index = self._get_features_batch(i)
        prediction = predictor_function(features)

        # get reference values
        batch_ref_values, batch_ref_index = self._get_decode_ref_values(i, [col for col, _ in decoding_functions])

        # finally we can decode our prediction to shape (batch_size, features, lstm_hist, aggregation)
        decoded_batch = np.stack([self._decode(sample, batch_ref_values[i], decoding_functions)
                                  for i, sample in enumerate(prediction)], axis=1)

        return decoded_batch, index, batch_ref_values, batch_ref_index

    def _get_decode_ref_values(self, i, columns, is_lstm_return_sequence=False):
        ref_loc = self._get_end_of_features_loc(i)
        ref_values, ref_index = self._get_reference_values(ref_loc, columns)

        # now we need to reshape the reference values to fit the batch and lstm sizes
        if is_lstm_return_sequence:
            batch_ref_values = [[ref_values[col, i:i + self.lstm_memory_size]
                                 for col in range(len(columns))]
                                for i in range(self.batch_size)]
        else:
            batch_ref_values = [[ref_values[col, i + self.lstm_memory_size - 1:i + self.lstm_memory_size]
                                 for col in range(len(columns))]
                                for i in range(self.batch_size)]

        batch_ref_index = [ref_index[i + self.lstm_memory_size - 1] for i in range(self.batch_size)]
        return np.stack(batch_ref_values, axis=0), batch_ref_index

    def _decode(self, vector, reference_values, decoding_functions):
        # first reshape a 1D vector into (nr of labels, 1, aggregation window)
        # fixme the hard coded 1 is wrong if self.return_sequences
        decoded = vector.reshape((len(self.labels), 1, -1))

        # now we can decode each row of each column with its associated decoder
        decoded = np.stack([np.array([func(decoded[i, row], reference_values[i, row], False)
                                      for row in (range(decoded.shape[1]))])
                            for i, (col, func) in enumerate(decoding_functions)], axis=0)

        return decoded

    @staticmethod
    def _get_column_names(encoders):
        return [col for i, (col, _) in enumerate(encoders)]

    def _get_last_features(self, n=-1):
        i = self.predictive_length() + n
        features_batch, index = self._get_features_batch(i)
        return features_batch[-1], index[-1]

    def predictive_length(self):
        return len(self) + self.forecast_horizon

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            raise ValueError('Shuffling not implemented yet')


class TestDataGenerator(AbstractDataGenerator):

    def __init__(self, data_generator, training_percentage: float = None):
        super(TestDataGenerator, self).__init__(data_generator.dataframe,
                                                data_generator.features,
                                                data_generator.labels,
                                                data_generator.batch_size,
                                                data_generator.lstm_memory_size,
                                                data_generator.aggregation_window_size,
                                                data_generator.forecast_horizon,
                                                training_percentage or data_generator.training_percentage,
                                                data_generator.return_sequences,
                                                data_generator.variances,
                                                True)

    def on_epoch_end(self):
        pass


class PredictiveDataGenerator(AbstractDataGenerator):

    # we need to extend DataGenerator because of windowing, encoding and decoding ...
    def __init__(self, model_path: str, data_fetcher: pd.DataFrame):  # FIXME accept DataFetcher only
        self.model_file_name = os.path.join(model_path, 'model')

        # load all parameters but the dataframe form a file
        with io.open(self.model_file_name + '.dg', 'rb') as f:
            pickles = cloudpickle.load(f)

        super(PredictiveDataGenerator, self).__init__(data_fetcher,  # getDataFrame
                                                      pickles[3],   # features
                                                      pickles[4],   # labels
                                                      pickles[5],   # batch_size
                                                      pickles[6],   # lstm_memory_size
                                                      pickles[7],   # aggregation_window_size
                                                      pickles[8],   # forecast_horizon
                                                      1.0,
                                                      pickles[9],   # return_sequences
                                                      pickles[10],  # variances
                                                      False)

        self.model = load_model(self.model_file_name + '.h5')
        self.epoch_hist = pickles[0]  # epoch_hist
        self.batch_hist = pickles[1]  # batch_hist

        column_names, predictions, reference_values, reference_index, labels, r_squares, self.standard_deviations, confidence = pickles[2]  # back_test
        self.back_test_history: BackTestHistory = BackTestHistory(column_names,
                                                                  predictions,
                                                                  reference_values,
                                                                  reference_index,
                                                                  labels, r_squares,
                                                                  self.standard_deviations,
                                                                  confidence)

    # TODO we might fix the backtest metod to provide the model.predict as default
    # def back_test(self, batch_predictor: Callable[[np.ndarray], np.ndarray] = None):
    #    return super(PredictiveDataGenerator, self).back_test(batch_predictor or self.model.predict)

    def predict(self, i: int = -1, confidence=.80):
        prediction, index, ref_values, ref_index = self._predict(self.model.predict, i)

        # get all the past data from the data frame from past lstm_memorysize rows
        columns = self._get_column_names(self.labels)
        loc_end = self.dataframe.index.get_loc(ref_index) + 1
        loc_start = max(0, loc_end - self.lstm_memory_size)
        df_past = self.dataframe[columns][loc_start:loc_end]

        # make a data-frame for the prediction
        z = get_std_confidence_factor(confidence)
        y_hat = {f'predicted_{col}': prediction[i, -1] for i, col in enumerate(columns)}
        lower = {f'predicted_{col}_lower': prediction[i, -1] - self.standard_deviations * z for i, col in enumerate(columns)}
        upper = {f'predicted_{col}_upper': prediction[i, -1] + self.standard_deviations * z for i, col in enumerate(columns)}
        df_predict = pd.DataFrame({**y_hat, **lower, **upper},
                                  index=pd.date_range(start=df_past.index[-1], periods=prediction.shape[-1] + 1)[1:])

        # concat the past and prediction
        prediction_df = pd.concat([df_past, df_predict], sort=True)
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
                 batch_size: int = 100, lstm_memory_size: int = 52 * 5, aggregation_window_size: int = 32, forecast_horizon: int = None,
                 training_percentage: float = 0.8,
                 return_sequences: bool = False,
                 variances: Dict[str, float] = {".*": 0.94},
                 model_path: str = "{}/{}-{}".format(tempfile.gettempdir(), datetime.now().strftime("%Y-%m-%d-%H-%M-%S"), str(uuid.uuid4()))):

        # we need to resolve the regular expressions and therefore we already need to know the additional columns
        # added by the super class. this is the sinusoidal time as well as the variances
        expanded_variances = [(col, r) for col in dataframe.columns for f, r in variances.items() if re.search(f, col)]
        columns = list(dataframe.columns.values) + \
                  [col + "_variance" for col, _ in expanded_variances] + \
                  ["trigonometric_time." + sin_time for sin_time in sinusoidal_time_calculators.keys()]

        super(DataGenerator, self).__init__(dataframe,
                                            [(col, r) for col in columns for f, r in features.items() if re.search(f, col)],
                                            [(col, r) for col in columns for l, r in labels.items() if re.search(l, col)],
                                            batch_size,
                                            lstm_memory_size,
                                            aggregation_window_size,
                                            aggregation_window_size if forecast_horizon is None else forecast_horizon,
                                            training_percentage,
                                            return_sequences,
                                            expanded_variances,
                                            False)

        super(DataGenerator, self).on_epoch_end()

        # make directories and file name
        os.makedirs(model_path, exist_ok=True)
        self.model_path = model_path
        self.model_filename = os.path.join(model_path, "model")

    def as_test_data_generator(self, training_percentage: float = None) -> TestDataGenerator:
        return TestDataGenerator(self, training_percentage)

    def get_max_batch_size(self):
        return math.gcd(len(self), len(self.as_test_data_generator()))

    def fit(self,
            model: keras.Model,
            fit_generator_args: Dict,
            quality_measure: Callable[[np.ndarray, np.ndarray, np.ndarray], float] = get_r_squared,
            predict_labels: bool = True,
            log_frequency: int = 50,
            log_dir: str = None) -> PredictiveDataGenerator:

        test_data = self.as_test_data_generator()

        callbacks = [
            BatchHistory()
            # FIXME callback = RelativeAccuracy(test_data, relative_accuracy_function, frequency, log_dir or self.model_path + logs)
        ]

        fit_generator_args["generator"] = self
        fit_generator_args["validation_data"] = test_data
        fit_generator_args["callbacks"] = callbacks + fit_generator_args.get("callbacks", [])

        # train the neural network using keras
        hist = model.fit_generator(**fit_generator_args)

        # convert history callbacks into data frames
        epoch_hist = pd.DataFrame(hist.history)
        batch_hist = pd.DataFrame(callbacks[0].history)

        # save keras model and weights
        logging.info(f'save keras model {self.model_filename }.h5')
        model.save(self.model_filename + '.h5')

        # backtest model to get some statistics
        back_test = test_data.back_test(model.predict, quality_measure=quality_measure, predict_labels=predict_labels)

        # save all data needed to do predictions and to reproduce the back test
        logging.info(f'save data model {self.model_filename }.dg')
        with io.open(self.model_filename + '.dg', 'wb') as f:
            cloudpickle.dump([
                epoch_hist,
                batch_hist,
                back_test.get_all_fields(),
                self.features,
                self.labels,
                self.batch_size,
                self.lstm_memory_size,
                self.aggregation_window_size,
                self.forecast_horizon,
                self.return_sequences,
                self.variances
            ], f)

        # return a re-usable predictive data generator
        return PredictiveDataGenerator(self.model_path, test_data.dataframe)
