from fastdtw import fastdtw
from keras.layers import LSTM
from keras.models import Sequential, load_model

from timeseries_ml_utils.data import *
from timeseries_ml_utils.statistics import *
from timeseries_ml_utils.encoders import *

if __name__ == "__main__":
    # fetch data
    data = DataFetcher(["GLD.US"], limit=500)
    data.fetch_data().tail()

    print(len(data.get_dataframe()))
    model_data = DataGenerator(data.get_dataframe(), {"^trigonometric": identity, "(Open|High|Low|Close)$": normalize},
                               {"GLD.US.Close$": normalize},
                               aggregation_window_size=16, batch_size=10,
                               model_filename="./test/resources/test-prediction-model-w16-b10-h260.h5")

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
                  "shuffle": False}

    # model_data.fit(model, train_args, frequency=10, relative_accuracy_function=relative_dtw, log_dir="/tmp/foo.123")

    predictor = model_data.as_predictive_data_generator()
    predictor._get_last_features()
    predicted_df = predictor.predict(-1)
    print(predicted_df)


    # train_data = model_data
    # test_data = model_data.as_test_data_generator()
    # callback = test_data.get_keras_callback(frequency=10, relative_accuracy_function=relative_dtw)
    # callback.clear_all_logs()
    #
    # hist = model.fit_generator(generator=train_data,
    #                            validation_data=test_data,
    #                            epochs=1,
    #                            use_multiprocessing=True,
    #                            workers=4,
    #                            shuffle=False,
    #                            callbacks=[callback])
