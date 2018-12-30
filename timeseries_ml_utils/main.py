
if __name__ == "__main__":
    from keras.layers import *
    from keras.models import *
    from timeseries_ml_utils.data import *
    from timeseries_ml_utils.statistics import *
    from timeseries_ml_utils.encoders import *
    from scipy.fftpack import dct, idct

    import matplotlib.pyplot as plt

    # fetch data
    data = DataFetcher(["GLD.US"], limit=350)  # 550
    # data = DataFetcher(["GLD.US", "DIA.US", "SHV.US", "TLT.US", "UUP.US", "IYT.US", "IYR.US"])

    print(data.fetch_data().tail())
    print(len(data.get_dataframe()))


    def cosine_transform_label(y, ref, encode):
        if encode:
            y = normalize(y, ref, True)
            return dct(y)[:len(y) // 2]
        else:
            return normalize(idct(np.hstack([y, np.zeros(len(y // 2))])), ref, False)


    def cosine_transform_feature(y, ref, encode):
        if encode:
            y = normalize(y, ref, True)
            return dct(y)
        else:
            return y


    model_data = DataGenerator(data.get_dataframe(),
                               {  # "^trigonometric": identity,
                                   # "Close_variance$": identity,
                                   # "(Open|High|Low|Close)$": normalize},
                                   # "(Close|Volume)$": cosine_transform_feature},
                                   "GLD.US.Close$": cosine_transform_feature},
                               {"GLD.US.Close$": cosine_transform_label},
                               aggregation_window_size=16, batch_size=10, forecast_horizon=8,
                               training_percentage=0.8,
                               model_path="/tmp/keras-regression-line-GLD-price-dct")
    print(model_data.get_df_columns())
    print("feature shape:", model_data.batch_feature_shape)
    print("labels shape:", model_data.batch_label_shape)
    print("train/test data:", len(model_data), len(model_data.as_test_data_generator()))
    print("max batch size:", model_data.get_max_batch_size())
    model_data.features, model_data.labels

    model = Sequential(name="LSTM-Model-IN")
    model.add(LSTM(model_data.batch_label_shape[-1],
                   name="LSTM-Layer-1",
                   batch_input_shape=model_data.batch_feature_shape,
                   activation='tanh',
                   dropout=0,
                   recurrent_dropout=0,
                   stateful=True,
                   return_sequences=True))

    model.add(LSTM(model_data.batch_label_shape[-1] * 2,
                   name="LSTM-Layer-2",
                   activation='tanh',
                   dropout=0,
                   recurrent_dropout=0,
                   stateful=True,
                   return_sequences=True))

    model.add(LSTM(int(model_data.batch_label_shape[-1] * 1.3),
                   name="LSTM-Layer-3",
                   activation='tanh',
                   dropout=0,
                   recurrent_dropout=0,
                   stateful=True,
                   return_sequences=True))

    model.add(LSTM(model_data.batch_label_shape[-1],
                   name="LSTM-Layer-4",
                   activation='tanh',
                   dropout=0,
                   recurrent_dropout=0,
                   stateful=True,
                   return_sequences=model_data.return_sequences))

    # model.add(TimeDistributed(Dense(model_data.batch_label_shape[-1],
    #                                activation='tanh',
    #                                name='TD-Dense-OUT')))

    model.compile("Adam", loss="mse", metrics=['mae', 'acc'])

    train_args = {"epochs": 1,
                  "use_multiprocessing": True,
                  "workers": 4,
                  "shuffle": False,
                  "verbose": 1,
                  # "callbacks": [EarlyStopping(monitor='val_acc', min_delta=0.01, mode='max', restore_best_weights=True)]
                  }

    fit = model_data.fit(model, train_args)
