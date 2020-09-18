import numpy as np


def prediction_interval(accuracy, z, num_samples):
    interval = z * np.sqrt((accuracy * (1 - accuracy)) / num_samples)
    lower = accuracy - interval
    upper = accuracy + interval
    return interval, lower, upper


def regression_interval(actual_probs, pred_probs, z):
    sum_errs = np.sum((actual_probs - pred_probs) ** 2)
    stdev = np.sqrt(1 / (len(actual_probs) - 2) * sum_errs)
    conf = z * stdev
    return conf
