from timeseries_ml_utils.tensorboard import TensorboardLogger
from timeseries_ml_utils.data import TestDataGenerator
from keras.callbacks import Callback
from keras import backend as K
from datetime import datetime
from fastdtw import fastdtw
from typing import Callable
import itertools as it
import numpy as np


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
                 frequency: int=50,
                 log_dir="/tmp/tb.log"):
        self.data_generator = data_generator
        self.relative_accuracy_function = relative_accuracy_function
        self.frequency = frequency
        self.tensorboard = TensorboardLogger(log_dir + "/{}".format(datetime.now().time()))
        self.model = None
        self.params = None
        self.r2 = None
        self.worst_sample = None
        self.log_step = it.count()

    def set_model(self, model):
        self.model = model
        if K.backend() == 'tensorflow':
            self.sess = K.get_session()

    def on_batch_end(self, batch, logs=None):
        if self.frequency > 0 and batch > 0 and batch % self.frequency == 0:
            self.r2, _ = self._relative_accuracy()
            self._log_histogram()

    def on_epoch_end(self, epoch, logs=None):
        self.r2, _ = self._relative_accuracy()
        self._log_histogram()

    def on_train_end(self, logs=None):
        self.r2, compare = self._relative_accuracy()

        # plot lne charts to tensorflow
        self.tensorboard.log_plots("bad", compare[:1])

        # TODO calculte the standard deviation of each predicted label

        # finally print a text representation of the histogram
        ascii_hist(self.r2, 10)
        # FIXME generate a line plot for the predictions and labels at the 80% confidence interval to tensorboard
        # FIXME generate a line plot for the predictions and labels at the low end to tensorboard

    def _relative_accuracy(self):
        batch_size = self.data_generator.batch_size
        r2 = np.array([])
        compare = []

        for i in range(0, len(self.data_generator), batch_size):
            features, labels = self.data_generator.__getitem__(i)
            prediction = self.model.predict(features, batch_size=batch_size)

            # calculate some kind of r² measure for each (label, prediction)
            r2_batch = np.array([self.relative_accuracy_function(labels[j], prediction[j]) for j in range(len(prediction))])

            # get a sorted index to pick samples of certain quality
            idx = np.argsort(self.r2)

            # FIXME only if len(shape) == 2
            # FIXME labels + prediction is a full batch
            compare.append(np.array([prediction[idx[int(len(idx) * 0.05)]], labels[idx[int(len(idx) * 0.05)]],
                                     prediction[idx[len(idx) // 2]], labels[idx[len(idx) // 2]],
                                     prediction[idx[int(len(idx) * 0.95)]], labels[idx[int(len(idx) * 0.95)]]]))

            # get an array of distances over all samples of all batches
            r2 = np.hstack([r2, r2_batch])

        # TODO calculate the standard deviation of each predicted label

        # now we have the similarity for each sample in all batches
        return r2, np.array(compare)

    def _log_histogram(self):
        print('\n', np.histogram(self.r2)[1], '\n')
        self.tensorboard.log_histogram("relative_accuracy", self.r2, next(self.log_step))

