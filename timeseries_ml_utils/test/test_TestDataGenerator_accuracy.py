from unittest import TestCase

from ..data import DataGenerator
from keras.models import Sequential
from keras.layers import LSTM
import pandas as pd
import numpy as np


class TestTestDataGenerator(TestCase):
    def test_on_epoch_end(self):
        # prepare data
        df = pd.DataFrame({
            "GLD.US.Close": np.arange(19.0),
            "GLD.US.Volume": np.arange(19.0)
        }, index=pd.date_range(start='01.01.2015', periods=19))


        # test case
        data_generator = DataGenerator(df, {"GLD.US.Close$": False, "GLD.US.Volume$": False}, {"GLD.US.Close$": False},
                                       3, 4, 5, training_percentage=0.6, return_sequences=False)

        model = Sequential()
        model.add(LSTM(data_generator.batch_label_shape[-1],
                       batch_input_shape=data_generator.batch_feature_shape,
                       activation='tanh',
                       dropout=0,
                       recurrent_dropout=0,
                       stateful=True,
                       return_sequences=data_generator.return_sequences))

        model.compile("Adam", loss="mse")

        test_dg = data_generator.as_test_data_generator(model)
        test_dg.on_epoch_end()
        print(test_dg.accuracy.tail())

        self.assertTrue(True)
