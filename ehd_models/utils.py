# -*- coding: utf-8 -*-
"""
Utilities for ML models.

Created on June 22 2022

@author: Oliver Nakano-Baker
"""
import numpy as np
from sklearn import metrics
from scipy.stats import pearsonr


def first_N_data_split(dataset, train_size, test_size):
    """Dataset is a dictionary. Create train and test dictionaries with the
    same keys. train_set gets the first train_size values, test_set gets the
    next test_size values."""
    return random_N_data_split(dataset, train_size, test_size, first=True)
    # train_set = {}
    # test_set = {}
    # for key, data in dataset.items():
    #     train_set[key] = data[:train_size]
    #     test_set[key] = data[train_size:train_size + test_size]
    # return train_set, test_set


def random_N_data_split(dataset, train_size, test_size, first=False):
    """Dataset is a dictionary. Create train and test dictionaries with the
    same keys. train_set gets the first train_size values, test_set gets the
    next test_size values."""
    train_set = {}
    test_set = {}

    train_idx = np.arange(train_size)
    test_idx = np.arange(test_size) + train_size
    if not first:               # Offset by a random allowed amount
        max_offset = len(dataset['X']) - (train_size+test_size) + 1
        try:
            offset = np.random.randint(low=0, high=max_offset)
        except:
            import ipdb; ipdb.set_trace()
        train_idx += offset
        test_idx += offset
    
    for key, data in dataset.items():
        train_set[key] = data[train_idx]
        test_set[key] = data[test_idx]
    return train_set, test_set


def dict_mean(dictionaries):
    """dictionaries: array of dictionaries.
    Return a dictionary where each key is drawn from the first element of the
    input and contains the mean of all values corresponding that key in the
    array."""
    agg = {}
    for key in dictionaries[0]:
        vals = np.array([d[key] for d in dictionaries]).astype(float)
        if not all(np.isnan(vals)):
            agg[key] =  np.nanmean(vals[vals != None])
        else:
            agg[key] = None
    return agg


def safe_pearson(x, y):
    if (y.max() - y.min()) == 0:
        return 0
    return pearsonr(x, y)[0]


def regression_metrics(truth, prediction):
    """Return a dictionary of metrics: MAE, MSE, MSLE, and MAPE"""
    result = {
        'MAE' : metrics.mean_absolute_error(truth, prediction),
        'MAPE': metrics.mean_absolute_percentage_error(truth[truth > 0],
                                                       prediction[truth > 0]),
        'MSE' : metrics.mean_squared_error(truth, prediction),
        # 'MSLE': metrics.mean_squared_log_error(truth, prediction),
        'r': safe_pearson(truth, prediction)
        }
    return result


def safe_auc(truth, prediction):
    try:
        return metrics.roc_auc_score(truth, prediction)
    except:
        return None


def classification_metrics(truth, prediction):
    """Return a dictionary of metrics: precision, recall, F1, AUC"""
    # For multi-classes or columns, just report on the first
    if len(truth.shape) > 1:
        t = truth[:, 0]
        p = prediction[:, 0]
    else:
        t = truth
        p = prediction

    result = {
        'class 1 precision': metrics.precision_score(t, p, zero_division=0),
        'class 1 recall'   : metrics.recall_score(t, p),
        'F1'       : metrics.f1_score(truth, prediction, average='micro'),
        'AUC'      : safe_auc(truth, prediction)
        }
    return result
