import os
from datetime import datetime
from shutil import rmtree
from unittest import TestCase
from ..tensorboard import TensorboardLogger
import numpy as np


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
