import os
from datetime import datetime
from shutil import rmtree
from unittest import TestCase
from ..tensorboard import TensorboardLogger
import numpy as np
import matplotlib.pyplot as plt
import itertools as it


class TestLogger(TestCase):

    def test_log_histogram(self):
        path = "/tmp/foo.123/{}".format(datetime.now().time())
        tb = TensorboardLogger(path)

        for i in range(1000):
            tb.log_histogram('test_hist_1', np.random.rand(50) * (i + 1), i)

        for i in range(1000):
            tb.log_histogram('test_hist_2', np.random.rand(50) * (i + 1), i)

        for i in range(1000):
            tb.log_histogram('test_hist_3', np.random.rand(50) * (i + 1), i)

        self.assertTrue(os.path.exists(path))
        rmtree(path)

    def test_log_plot(self):
        path = "/tmp/foo.123/{}".format(datetime.now().time())
        tb = TensorboardLogger(path)
        cnt = it.count()

        x = [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]]
        y = [[1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [7, 8, 9, 10]]
        colours = ['r', 'g', 'b', 'k']
        plots = []

        for i in range(len(x)):
            if i % 2 == 0:
                fig = plt.figure()

            plt.plot(x[i], y[i], colours[i])

            if i % 2 != 0:
                plots.append(fig)

        tb.log_plots("batch1", plots, next(cnt))
        self.assertTrue(os.path.exists(path))
        rmtree(path)
