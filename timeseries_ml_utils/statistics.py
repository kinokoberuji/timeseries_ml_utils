from timeseries_ml_utils.data import TestDataGenerator
from keras.callbacks import Callback
from keras import backend as K
from typing import Callable
from fastdtw import fastdtw
import numpy as np
import asciichartpy as ascii


def relative_dtw(x, y):
    x1 = x + 1
    y1 = y + 1
    prediction_distance = fastdtw(x1, y1)[0]
    max_dist = max(len(y) * x1.max(), len(y) * y1.max())
    return (max_dist - prediction_distance) / max_dist


def r_square(x, y):
    x1 = x + 1
    y1 = y + 1
    x1_bar = x1.mean()
    return 1 - np.sum((x1 - y1) ** 2) / np.sum((x1 - x1_bar) ** 2)


def relative_dtw_times_r2(x, y):
    return relative_dtw(x, y) * r_square(x, y)


def ascii_hist(x, bins):
    N, X = np.histogram(x, bins=bins)
    total = 1.0 * len(x)
    width = 50
    nmax = N.max()

    for (xi, n) in zip(X,N):
        bar = '#'*int(n*1.0*width/nmax)
        xi = '{0: <8.4g}'.format(xi).ljust(10)
        print('{0}| {1}'.format(xi, bar))


class RelativeAccuracy(Callback):
    # see: https://keunwoochoi.wordpress.com/2016/07/16/keras-callbacks/

    def __init__(self,
                 data_generator: TestDataGenerator,
                 relative_accuracy_function: Callable[[np.ndarray, np.ndarray], np.ndarray]=relative_dtw,
                 frequency: int=50):
        self.data_generator = data_generator
        self.relative_accuracy_function = relative_accuracy_function
        self.frequency = frequency
        self.model = None
        self.params = None
        self.r2 = None
        self.worst_sample = None

    def set_model(self, model):
        self.model = model
        if K.backend() == 'tensorflow':
            self.sess = K.get_session()

    def on_batch_end(self, batch, logs=None):
        if self.frequency > 0 and batch % self.frequency == 0:
            self.r2, _ = self.relative_accuracy()
            print('\n', np.histogram(self.r2)[1], '\n')
            # TODO i would like to tensorboard print it here like so: https://stackoverflow.com/a/48876774/1298461

    def on_epoch_end(self, epoch, logs=None):
        self.r2, _ = self.relative_accuracy()
        print('\n', np.histogram(self.r2)[1], '\n')

    def on_train_end(self, logs=None):
        self.r2, self.worst_sample = self.relative_accuracy()
        ascii_hist(self.r2, 10)
        # ascii.plot()

    def relative_accuracy(self):
        r2 = np.array([])
        r2_batch = np.array([])
        batch_size = self.data_generator.batch_size

        for i in range(0, len(self.data_generator), batch_size):
            features, labels = self.data_generator.__getitem__(i)
            prediction = self.model.predict(features, batch_size=batch_size)

            # calculate some kind of r² measure for each (label, prediction)
            r2_batch = np.array([self.relative_accuracy_function(labels[j], prediction[j]) for j in range(len(prediction))])
            r2 = np.hstack([r2, r2_batch])

        min_index = np.argmin(r2_batch)
        max_index = np.argmax(r2_batch)

        # now we have the similarity for each sample in all batches
        return r2, {"label_" + str(r2_batch.min()): labels[min_index],
                    "prediction_" + str(r2_batch.min()): prediction[min_index],
                    "label_" + str(r2_batch.max()): labels[max_index],
                    "prediction_" + str(r2_batch.max()): prediction[max_index]
                    }

