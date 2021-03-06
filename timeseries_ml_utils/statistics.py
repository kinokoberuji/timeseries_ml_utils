from fastdtw import fastdtw
from scipy.fftpack import dct
import math
import re
import pandas as pd
import numpy as np


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
