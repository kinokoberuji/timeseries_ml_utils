from timeseries_ml_utils.data import TestDataGenerator
from keras.callbacks import Callback
from typing import Callable
from fastdtw import fastdtw
import numpy as np


def relative_dtw(x, y):
    prediction_distance = fastdtw(x, y)[0]
    max_dist = len(y) * x.max()
    return (max_dist - prediction_distance) / max_dist


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

    def set_model(self, model):
        self.model = model

    def on_batch_end(self, batch, logs=None):
        if self.frequency > 0 and batch % self.frequency == 0:
            self.r2 = self.relative_accuracy()
            print('\n', np.histogram(self.r2)[1], '\n')
            # TODO i would like to tensorboard print it here

    def on_epoch_end(self, epoch, logs=None):
        self.r2 = self.relative_accuracy()
        # TODO store one label and prediction array for each bin so that we can plot it against each other
        ascii_hist(self.r2, 10)

    def relative_accuracy(self):
        r2 = np.array([])
        batch_size = self.data_generator.batch_size
        for i in range(0, len(self.data_generator), batch_size):
            features, labels = self.data_generator.__getitem__(i)
            prediction = self.model.predict(features, batch_size=batch_size)

            # calculate some kind of r² measure for each (label, prediction)
            r2 = np.hstack([r2, [self.relative_accuracy_function(labels[j], prediction[j])
                                 for j in range(len(prediction))]])

        # now we have the similarity for each sample in all batches
        return r2


