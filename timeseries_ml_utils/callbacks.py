import itertools as it
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
            self.r2, _ = self._relative_accuracy()
            self._log_histogram()

        for tag, val in logs.items():
            self.tensorboard.log_scalar(tag, val, next(self.log_scalar_step))

    def on_epoch_end(self, epoch, logs=None):
        self.r2, _ = self._relative_accuracy()
        self._log_histogram()

    def on_train_end(self, logs=None):
        self.r2, samples = self._relative_accuracy()

        # generate line plots for the predictions vs labels
        self.tensorboard.log_plots("bad fits 0.05", [RelativeAccuracy._plot(batch[0], batch[1]) for batch in samples], 0)
        self.tensorboard.log_plots("average fits", [RelativeAccuracy._plot(batch[2], batch[3]) for batch in samples], 0)
        self.tensorboard.log_plots("best fits 0.95", [RelativeAccuracy._plot(batch[4], batch[5]) for batch in samples], 0)

        # finally print a text representation of the histogram
        print("Tested {} samples".format(len(samples)))
        ascii_hist(self.r2, 10)

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
            idx = np.argsort(r2_batch)

            if len(prediction.shape) == 2:
                compare.append(np.array([prediction[idx[int(len(idx) * 0.05)]], labels[idx[int(len(idx) * 0.05)]],
                                         prediction[idx[len(idx) // 2]], labels[idx[len(idx) // 2]],
                                         prediction[idx[int(len(idx) * 0.95)]], labels[idx[int(len(idx) * 0.95)]]]))

            # get an array of distances over all samples of all batches
            r2 = np.hstack([r2, r2_batch])

        # TODO calculate the standard deviation of each predicted label

        # now we have the similarity for each sample in all batches
        return r2, compare

    def _log_histogram(self):
        print('\n', np.histogram(self.r2)[1], '\n')
        self.tensorboard.log_histogram("relative_accuracy", self.r2, next(self.log_histogram_step))

    def clear_last_log(self):
        rmtree(self.log_dir, ignore_errors=True)

    def clear_all_logs(self):
        rmtree(self.log_dir_root, ignore_errors=True)

    @staticmethod
    def _plot(prediction, label):
        fig = plt.figure()
        plt.plot(prediction, label='predict')
        plt.plot(label, label='label')
        plt.legend(loc='best')
        return fig
