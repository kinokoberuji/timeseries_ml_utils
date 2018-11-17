from unittest import TestCase
import pandas as pd
import numpy as np

from ..data import DataGenerator

pd.options.display.max_columns = None


class TestDataGenerator(TestCase):

    def test___getitem__(self):
        pd.options.display.max_columns = None

        df = pd.DataFrame({
            "GLD.US.Close": np.arange(19.0),
            "GLD.US.Volume": np.arange(19.0)
        }, index=pd.date_range(start='01.01.2015', periods=19))

        data_generator = DataGenerator(df, ["GLD.US.Close$", "GLD.US.Volume$"], ["GLD.US.Close$"], 3, 4, 5, training_percentage=0.6, return_sequences=True)
        last_index = len(data_generator) - 1

        print("\n", data_generator.dataframe)

        first_batch = data_generator.__getitem__(0)

        # assert window aggregation
        self.assertEqual(data_generator.__aggregate_normalized_window__(last_index)[0][-1][-1][-1], 13.)
        self.assertEqual(data_generator.__aggregate_normalized_window__(last_index)[1][-1][-1][-1], 14.)

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