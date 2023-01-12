# -*- coding: utf-8 -*-
"""
Created on June 15 2022

@author: Oliver Nakano-Baker
"""
import numpy as np
import pandas as pd

from .mle_models import MLE_Regressor, MLE_Classifier
from .cold_models import (RF_Regressor, MLP_Regressor,
                          RF_Classifier, MLP_Classifier)
from .brute_pretrained_models import (RF_Regressor_Allpre,
                                      RF_Classifier_Allpre,
                                      MLP_Regressor_Allpre,
                                      MLP_Classifier_Allpre)
                                      # RF_Regressor_lastXY,
                                      # RF_Classifier_lastXY,
                                      # MLP_Regressor_lastXY,
                                      # MLP_Classifier_lastXY)

from .utils import random_N_data_split, dict_mean


def make_model_like(architecture, params):
    # Cold Start Regressors
    if architecture == 'MLE'\
       or (architecture == 'normed_MLE'):
        return MLE_Regressor(params)
    if architecture == 'cold_RF':
        return RF_Regressor(params)
    if architecture == 'cold_MLP':
        return MLP_Regressor(params)
    if (architecture == 'only_pretrained_RF')\
       or (architecture == 'normed_RF')\
       or (architecture == 'v_normed_RF'):
        return RF_Regressor_Allpre(params)
    if architecture == 'only_pretrained_MLP':
        return MLP_Regressor_Allpre(params)
    # if architecture == 'lastXY_RF':
    #     return RF_Regressor_lastXY(params)
    # if architecture == 'lastXY_MLP':
    #     return MLP_Regressor_lastXY(params)

    # Cold Start Classifiers
    if architecture == 'MLE_class'\
       or (architecture == 'normed_MLE_class'):
        return MLE_Classifier(params)
    if architecture == 'cold_RF_class':
        return RF_Classifier(params)
    if architecture == 'cold_MLP_class':
        return MLP_Classifier(params)
    if architecture == 'only_pretrained_RF_class'\
       or (architecture == 'normed_RF_class')\
       or (architecture == 'v_normed_RF_class'):
        return RF_Classifier_Allpre(params)
    if architecture == 'only_pretrained_MLP_class':
        return MLP_Classifier_Allpre(params)
    # if architecture == 'lastXY_RF_class':
    #     return RF_Classifier_lastXY(params)
    # if architecture == 'lastXY_MLP_class':
    #     return MLP_Classifier_lastXY(params)


    else:
        raise ValueError(f"Not a valid model architecture: {architecture}")


class EHD_Model:
    log_slices = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000,
                           5000, 10_000])
    def __init__(self, architecture, params={}):
        self.architecture = architecture
        self.model = make_model_like(architecture, params)
        self.pretrainer = self.model.pretrainer

    def pretrain(self, dataset):
        self.model.pretrain(dataset)

    def evaluate(self, dataset, train_sizes=None, test_size=20, samples_to_try=500):
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
            trials = []
            num_trials = np.ceil(7 * np.log(100 / train_size + 1)).astype(int)
            for _ in range(num_trials):
                # create train and test sets from the dataset
                train_set, test_set =  random_N_data_split(dataset, train_size,
                                                           test_size)
                # retrain self.model on the train set
                temp_model = self.model.copy()
                temp_model.retrain(train_set)
                result = temp_model.evaluate(test_set)
                result['train_size'] = train_size
                trials.append(result)

            results.append(dict_mean(trials))

        return pd.DataFrame.from_dict(results)

    @property
    def xtype(self):
        return self.model.xtype

    @property
    def ytype(self):
        return self.model.ytype

    @property
    def filters(self):
        return self.model.filters
