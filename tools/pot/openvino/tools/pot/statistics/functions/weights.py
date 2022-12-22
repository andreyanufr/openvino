# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from functools import partial

import numpy as np
from bisect import bisect_left

from ..function_selector import WEIGHTS_STATS_FN, PERTENSOR, PERCHANNEL

w_stats_fn_per_tensor = WEIGHTS_STATS_FN[PERTENSOR]
w_stats_fn_per_channel = WEIGHTS_STATS_FN[PERCHANNEL]


# helper functions to calculate per-filter statistics for weights
def calculate_per_filter_stats(weights, fn, transpose=False):
    """ Calculates per-filter statistics for weights using a specific function
    :param weights: model layer weights
    :param fn: function to calculate per-filter statistics
    :param transpose: transpose weights data from IOHW to OIHW to collect stats
    :return statistics generated by fn
    """
    if transpose:
        weights_shape = [1, 0]
        original_axes = np.array(range(len(weights.shape)))
        weights_shape.extend(original_axes[2:])
        weights = np.transpose(weights, weights_shape)
    if not weights.shape:
        return fn(weights)
    t = np.reshape(weights, (weights.shape[0], -1))
    return fn(t, axis=1)


@w_stats_fn_per_tensor.register('max')
def max_per_tensor(weights):
    return np.max(weights)


@w_stats_fn_per_tensor.register('min')
def min_per_tensor(weights):
    return np.min(weights)


@w_stats_fn_per_tensor.register('abs_max')
def abs_max_per_tensor(weights):
    return np.max(np.abs(weights))


@w_stats_fn_per_tensor.register('quantile')
def quantile_per_tensor(weights, q):
    return np.quantile(weights, q=q)


@w_stats_fn_per_tensor.register('abs_quantile')
def abs_quantile_per_tensor(weights, q):
    return np.quantile(np.abs(weights), q=q)


@w_stats_fn_per_channel.register('max')
def max_per_filter(weights, transpose=False):
    return calculate_per_filter_stats(weights, np.max, transpose=transpose)


@w_stats_fn_per_channel.register('min')
def min_per_filter(weights, transpose=False):
    return calculate_per_filter_stats(weights, np.min, transpose=transpose)


@w_stats_fn_per_channel.register('abs_max')
def abs_max_per_filter(weights, transpose=False):
    return max_per_filter(np.abs(weights), transpose=transpose)


@w_stats_fn_per_channel.register('quantile')
def quantile_per_filter(weights, q, transpose=False):
    return calculate_per_filter_stats(weights, partial(np.quantile, q=q), transpose=transpose)


@w_stats_fn_per_channel.register('abs_quantile')
def abs_quantile_per_filter(weights, q, transpose=False):
    return quantile_per_filter(np.abs(weights), q, transpose=transpose)


def find_closest(arr, num):
    pos = bisect_left(arr, num)
    if pos == 0:
        return arr[0]
    if pos == len(arr):
        return arr[-1]
    before = arr[pos - 1]
    after = arr[pos]
    if after - num < num - before:
        return after
    else:
        return before


def find_closest_quantize(quants, data):
    res = [0] * len(data)
    for i, val in enumerate(data):
        res[i] = find_closest(quants, val)
    return res


def mse_scale_per_tensor(x):
    '''
    "FP8 Quantization: The Power of the Exponent" like MSE for weight for e1m3e4 with initial bias 11
    x - tensor
    '''
    hf8_abs_quants = [0.0, 0.00012207, 0.000244141, 0.000366211, 0.000488281,
                     0.000610352, 0.000732422, 0.000854492, 0.000976562, 0.00109863, 0.0012207, 0.00134277, 0.00146484, 0.00158691,
                     0.00170898, 0.00183105, 0.00195312, 0.00219727, 0.00244141, 0.00268555, 0.00292969, 0.00317383, 0.00341797,
                     0.00366211, 0.00390625, 0.00439453, 0.00488281, 0.00537109, 0.00585938, 0.00634766, 0.00683594, 0.00732422,
                     0.0078125, 0.00878906, 0.00976562, 0.0107422, 0.0117188, 0.0126953, 0.0136719, 0.0146484, 0.015625, 0.0175781,
                     0.0195312, 0.0214844, 0.0234375, 0.0253906, 0.0273438, 0.0292969, 0.03125, 0.0351562, 0.0390625, 0.0429688,
                     0.046875, 0.0507812, 0.0546875, 0.0585938, 0.0625, 0.0703125, 0.078125, 0.0859375, 0.09375, 0.101562, 0.109375,
                     0.117188, 0.125, 0.140625, 0.15625, 0.171875, 0.1875, 0.203125, 0.21875, 0.234375, 0.25, 0.28125, 0.3125, 0.34375,
                     0.375, 0.40625, 0.4375, 0.46875, 0.5, 0.5625, 0.625, 0.6875, 0.75, 0.8125, 0.875, 0.9375, 1.0, 1.125, 1.25, 1.375,
                     1.5, 1.625, 1.75, 1.875, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0,
                     9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0]

    x = np.abs(x).flatten()
    sigma = hf8_abs_quants[-1] / (np.max(x) + np.finfo(float).eps)
    sigma_min = 0.1 * sigma
    sigma_max = 1.2 * sigma
    n_steps = 100

    sigma_step = (sigma_max - sigma_min) / n_steps

    min_mse = np.finfo(float).max
    best_sigma = sigma
    sigma_cur = sigma_min

    while sigma_cur < sigma_max:
        scaled_data = sigma_cur * x
        y = find_closest_quantize(hf8_abs_quants, scaled_data)
        tmp = np.mean((y - scaled_data)**2)
        if tmp < min_mse:
            min_mse = tmp
            best_sigma = sigma_cur
    return best_sigma


def mse_scale_per_channel(x):
    res = [1.0] * x.shape[0]
    for i in range(x.shape[0]):
        res[i] = mse_scale_per_tensor(x[i, ...])

    res = np.array(res)

    return res


@w_stats_fn_per_channel.register('mse')
def mse_scale_per_channel(weights, transpose=False):
    return mse_scale_per_channel(weights)
