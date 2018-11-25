import itertools as it
import math
from datetime import datetime
from shutil import rmtree

import numpy as np
from keras import backend as K
from keras.callbacks import Callback
from matplotlib import pyplot as plt

from .statistics import ascii_hist, relative_dtw
from .tensorboard import TensorboardLogger


class RelativeAccuracy(Callback):
    # see: https://keunwoochoi.wordpress.com/2016/07/16/keras-callbacks/

    def __init__(self,
                 data_generator,
                 relative_accuracy_function=relative_dtw,
                 frequency=50,
                 log_dir="/tmp/tb.log"):
        self.data_generator = data_generator
        self.relative_accuracy_function = relative_accuracy_function
        self.frequency = frequency
        self.log_dir_root = log_dir
        self.log_dir = log_dir + "/{}".format(datetime.now().time())
        self.tensorboard = None
        self.model = None
        self.params = None
        self.r2 = None
        self.worst_sample = None
        self.sess = None
        self.log_histogram_step = it.count()
        self.log_scalar_step = it.count()

    def set_model(self, model):
        self.model = model
        if K.backend() == 'tensorflow':
            self.sess = K.get_session()

    def on_train_begin(self, logs=None):
        self.tensorboard = TensorboardLogger(self.log_dir)

    def on_batch_end(self, batch, logs=None):
        if self.frequency > 0 and batch > 0 and batch % self.frequency == 0:
            self.r2, _, _, _, _ = self._relative_accuracy()
            self._log_histogram()

        for tag, val in logs.items():
            self.tensorboard.log_scalar(tag, val, next(self.log_scalar_step))

    def on_epoch_end(self, epoch, logs=None):
        self.r2, _, _, _, _ = self._relative_accuracy()
        self._log_histogram()

    def on_train_end(self, logs=None):
        self.r2, features, labels, predictions, errors = self._relative_accuracy()

        sorted_index = np.argsort(self.r2)
        bad_idx = int(len(sorted_index) * 0.10)
        avg_idx = len(sorted_index) // 2
        top_idx = int(len(sorted_index) * 0.90)
        print("bad {}, avg {}, top {}, last {}".format(bad_idx, avg_idx, top_idx, len(sorted_index)))

        self.tensorboard.log_plots("bad fits <= 0.05",
                                   [RelativeAccuracy._plot(predictions[sorted_index[i]],
                                                           labels[sorted_index[i]],
                                                           self.r2[sorted_index[i]])
                                    for i in range(max(bad_idx - 10, 0), bad_idx)],
                                   0)

        self.tensorboard.log_plots("average fits ~ 0.5",
                                   [RelativeAccuracy._plot(predictions[sorted_index[i]],
                                                           labels[sorted_index[i]],
                                                           self.r2[sorted_index[i]])
                                    for i in range(max(avg_idx - 5, 0), min(avg_idx + 5, len(predictions)))],
                                   0)

        self.tensorboard.log_plots("top fits >= 0.95",
                                   [RelativeAccuracy._plot(predictions[sorted_index[i]],
                                                           labels[sorted_index[i]],
                                                           self.r2[sorted_index[i]])
                                    for i in range(top_idx, min(top_idx + 10, len(predictions)))],
                                   0)

        ascii_hist(self.r2, 10)

    def _relative_accuracy(self):
        batch_size = self.data_generator.batch_size
        predictions = []
        features = []
        labels = []
        r2 = []

        for i in range(0, len(self.data_generator), batch_size):
            features_batch, labels_batch = self.data_generator.__getitem__(i)
            predictions_batch = self.model.predict(features_batch, batch_size=batch_size)

            for j in range(batch_size):
                features.append(features_batch[j])
                labels.append(labels_batch[j])
                predictions.append(predictions_batch[j])
                r2.append(self.relative_accuracy_function(labels_batch[j], predictions_batch[j]))

        predictions = np.array(predictions)
        features = np.array(features)
        labels = np.array(labels)
        r2 = np.array(r2)

        errors = np.array(
            [math.sqrt(((labels[:, i] - predictions[:, i]) ** 2).sum() / (batch_size - 1)) for i in range(len(labels[0]))])

        return r2, features, labels, predictions, errors

    def _log_histogram(self):
        print('\n', np.histogram(self.r2)[1], '\n')
        self.tensorboard.log_histogram("relative_accuracy", self.r2, next(self.log_histogram_step))

    def clear_last_log(self):
        rmtree(self.log_dir, ignore_errors=True)

    def clear_all_logs(self):
        rmtree(self.log_dir_root, ignore_errors=True)

    @staticmethod
    def _plot(prediction, label, title):
        fig = plt.figure()
        plt.plot(prediction, label='predict')
        plt.plot(label, label='label')
        plt.legend(loc='best')
        plt.title("{}".format(title))
        return fig
