from io import BytesIO
from typing import List

import numpy as np
import matplotlib.pyplot as plt

class TensorboardLogger(object):
    """Logging in tensorboard without tensorflow ops."""

    def __init__(self, log_dir):
        """Creates a summary writer logging to log_dir."""
        global tf
        try:
            import tensorflow as tf

            self.writer = tf.summary.FileWriter(log_dir)
        except ImportError:
            raise ImportError('You need the TensorFlow module installed to '
                              'use TensorBoard.')

    def log_scalar(self, tag, value, step):
        """Log a scalar variable.

        Parameter
        ----------
        tag : basestring
            Name of the scalar
        value
        step : int
            training iteration
        """
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag,
                                                     simple_value=value)])
        self.writer.add_summary(summary, step)

    def log_plots(self, tag, figures: List[plt.Figure], step):
        """Logs a plot as image."""

        im_summaries = []
        for nr, fig in enumerate(figures):
            # Write the image to a string
            buf = BytesIO()
            fig.savefig(buf, format='png', quality=10)
            width, height = fig.get_size_inches() * fig.dpi

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=buf.getvalue(),
                                       height=int(height),
                                       width=int(width))
            # Create a Summary value
            im_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, nr),
                                                 image=img_sum))

            # close buffer
            buf.close()

        # Create and write Summary
        summary = tf.Summary(value=im_summaries)
        self.writer.add_summary(summary, step)

    def log_histogram(self, tag, values, step, bins=1000):
        """Logs the histogram of a list/vector of values."""
        # Convert to a numpy array
        values = np.array(values)

        # Create histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill fields of histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Requires equal number as bins, where the first goes from -DBL_MAX to bin_edges[1]
        # See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/framework/summary.proto#L30
        # Thus, we drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()
