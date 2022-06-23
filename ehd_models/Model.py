# -*- coding: utf-8 -*-
"""
Created on June 15 2022

@author: Oliver Nakano-Baker
"""
import numpy as np
import pandas as pd

from .mle_models import MLE_Regressor, MLE_Classifier
from .utils import first_N_data_split


def make_model_like(architecture, params):
    if architecture == 'MLE':
        return MLE_Regressor(params)
    if architecture == 'MLE_class':
        return MLE_Classifier(params)
    else:
        raise ValueError(f"Not a valid model architecture: {architecture}")


class EHD_Model:
    log_slices = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000,
                           5000, 10_000])
    def __init__(self, architecture, params={}):
        self.architecture = architecture
        self.model = make_model_like(architecture, params)

    def pretrain(self, dataset):
        self.model.pretrain(dataset)

    def evaluate(self, dataset, train_sizes=None, test_size=20):
        """Using copies of the model, update it with increasingly large
        training slices of the new dataset. The update function calls
        model.retrain(). Then evaluate the "retrained" model on the next
        test_size printed features."""

        if train_sizes is None:
            upper_limit = len(dataset['X']) - test_size
            train_sizes = self.log_slices[self.log_slices < upper_limit]
            train_sizes = np.concatenate((train_sizes, [upper_limit]))

        results = []
        for train_size in train_sizes:
            temp_model = self.model.copy()
            # create train and test sets from the dataset
            train_set, test_set =  first_N_data_split(dataset, train_size,
                                                      test_size)

            # retrain self.model on the train set
            temp_model.retrain(train_set)
            result = temp_model.evaluate(test_set)
            result['train_size'] = train_size
            results.append(result)

        return pd.DataFrame.from_dict(results)
