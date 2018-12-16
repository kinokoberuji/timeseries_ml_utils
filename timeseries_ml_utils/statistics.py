from typing import Union, Iterable

from pandas_ml import ConfusionMatrix
from scipy.fftpack import dct
from random import randint
from fastdtw import fastdtw
import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd
import numpy as np
import math
import re


def dct_distance(x, y):
    cos_x = dct(x)
    cos_y = dct(y)
    dist = 0

    for i in range(1, len(cos_x)):
        dist += abs(cos_x[i] - cos_y[i]) * (1 - 0.97) + 0.97 * dist

    return dist


def relative_dtw(x, y):
    x1 = x + 1
    y1 = y + 1
    prediction_distance = fastdtw(x1, y1)[0]
    max_dist = max(len(y) * x1.max(), len(y) * y1.max())
    return (max_dist - prediction_distance) / max_dist


def relative_dtw_2(x, y):
    prediction_distance = fastdtw(x, y, dist=2)[0]
    max_dist = len(y) * (np.abs(x).max() + np.abs(y).max())
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


def add_ewma_variance(df: pd.DataFrame, param: float):
    for rx, l in param.items():
        for col in df.columns:
            if re.search(rx, col):
                arr = df[col].pct_change().values
                all_var = []
                var = 0

                for i in range(len(arr)):
                    v = l * var + (1 - l) * arr[i] ** 2
                    var = 0 if math.isnan(v) or math.isinf(v) else v
                    all_var.append(var)

                df[col + "_variance"] = all_var

    return df


def add_sinusoidal_time(df):
    df["trigonometric_time.cos_dow"] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    df["trigonometric_time.sin_dow"] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df["trigonometric_time.cos_woy"] = np.cos(2 * np.pi * df.index.week / 52)
    df["trigonometric_time.sin_woy"] = np.sin(2 * np.pi * df.index.week / 52)
    df["trigonometric_time.cos_doy"] = np.cos(2 * np.pi * df.index.dayofyear / 366)
    df["trigonometric_time.sin_doy"] = np.sin(2 * np.pi * df.index.dayofyear / 366)
    df["trigonometric_time.sin_yer"] = np.sin(2 * np.pi * (df.index.year - (df.index.year // 10) * 10) / 9)
    df["trigonometric_time.cos_yer"] = np.cos(2 * np.pi * (df.index.year - (df.index.year // 10) * 10) / 9)
    df["trigonometric_time.sin_dec"] = np.sin(2 * np.pi * ((df.index.year - (df.index.year // 100) * 100) // 10) / 9)
    df["trigonometric_time.cos_dec"] = np.cos(2 * np.pi * ((df.index.year - (df.index.year // 100) * 100) // 10) / 9)
    return df


class BackTestHistory(object):

    def __init__(self, column_names, predictions, reference_values, labels, r_squares, standard_deviations,
                 confidence=.80):
        self.column_names = column_names
        self.predictions = predictions
        self.reference_values = reference_values
        self.labels = labels
        self.r_squares = r_squares
        self.standard_deviations = standard_deviations
        self.confidence = self._get_confidence_factor(confidence)

    def set_confidence(self, confidence=.80):
        self.confidence = self._get_confidence_factor(confidence)

    @staticmethod
    def _get_confidence_factor(confidence):
        return st.norm.ppf(confidence) if confidence is not None else 0

    def get_measures(self):
        return self.predictions, self.labels, self.r_squares, self.standard_deviations

    def confusion_matrix(self):
        nr_of_values = self.labels.shape[1]
        result = {}

        for i, label in enumerate(self.column_names):
            l = self.labels[i]
            p = self.predictions[i]
            r = self.reference_values[i]

            # one way would be to count positive returns
            y = [sum([f > 0 for f in l[j, -1] / r[j] - 1]) for j in range(nr_of_values)]
            y_hat = [sum([f > 0 for f in p[j, -1] / r[j] - 1]) for j in range(nr_of_values)]

            # another way would be bucketing

            # finally calculate the confusion matrix
            result[label] = ConfusionMatrix(y, y_hat)

        return result

    def hist(self, bins: Union[int, Iterable, str] = 10):
        return {label: np.histogram(self.r_squares[i, :, -1], bins) for i, label in enumerate(self.column_names)}

    def back_test_confidence(self):
        # TODO back test how often a label exceeded the confidence
        pass

    def plot_hist(self, figsize=None):
        fig = plt.figure(figsize=figsize)

        for i, label in enumerate(self.column_names):
            plt.hist(self.r_squares[i, :, -1], label=label)

        plt.legend(loc='best')
        plt.title('r²')
        return fig

    def plot_random_sample(self, figsize=None, loc=None):
        j = loc if loc is not None else randint(0, self.predictions.shape[1] - 1)
        fig = plt.figure(figsize=figsize)

        for i, label in enumerate(self.column_names):
            y = self.labels[i, j, -1]
            y_hat = self.predictions[i, j, -1]

            if self.confidence > 0:
                upper = y_hat + self.confidence * self.standard_deviations
                lower = y_hat - self.confidence * self.standard_deviations
                plt.fill_between(range(y_hat.shape[0]), upper, lower, alpha=.5)

            plt.plot(y_hat, label='predict')
            plt.plot(y, label='label')
            plt.legend(loc='best')
            plt.title("{}: {}, r²={:.2f}".format(j, label, self.r_squares[i, j, -1]))

        return fig

    def summary(self):
        # TODO return some summary like
        #  confusion matrix Phi coefficient
        #  r2 distribution
        #  some error measure
        #  and how often the error exceeds the expected error
        pass

