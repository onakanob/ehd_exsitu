# -*- coding: utf-8 -*-
"""
Utilities for ML models.

Created on June 22 2022

@author: Oliver Nakano-Baker
"""
from sklearn import metrics
from scipy.stats import pearsonr


def first_N_data_split(dataset, train_size, test_size):
    """Dataset is a dictionary. Create train and test dictionaries with the
    same keys. train_set gets the first train_size values, test_set gets the
    next test_size values."""
    train_set = {}
    test_set = {}
    for key, data in dataset.items():
        train_set[key] = data[:train_size]
        test_set[key] = data[train_size:train_size + test_size]
    return train_set, test_set


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
