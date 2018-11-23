from keras.layers import LSTM
from keras.models import Sequential, load_model

from timeseries_ml_utils.data import *
from timeseries_ml_utils.statistics import relative_dtw_times_r2

if __name__ == "__main__":
    # fetch data
    data = DataFetcher(["GLD.US"], limit=500)
    data.fetch_data().tail()

    print(len(data.get_dataframe()))
    model_data = DataGenerator(data.get_dataframe(), {"^trigonometric": False, "(Open|High|Low|Close)$": True}, {"GLD.US.Close$": True},
                               aggregation_window_size=16, batch_size=10)
    print(model_data.batch_feature_shape)
    print(model_data.batch_label_shape)
    model_data.features, model_data.labels

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
    print(model.layers[0].get_weights()[0][0])
    model.save("/tmp/keras-foo-1.h5")

    model = load_model("/tmp/keras-foo-1.h5")

    train_data = model_data
    test_data = model_data.as_test_data_generator(model)
    callback = test_data.get_keras_callback(frequency=10, relative_accuracy_function=relative_dtw_times_r2)
    callback.clear_all_logs()

    hist = model.fit_generator(generator=train_data,
                               validation_data=test_data,
                               epochs=1,
                               use_multiprocessing=True,
                               workers=4,
                               shuffle=False,
                               callbacks=[callback])
