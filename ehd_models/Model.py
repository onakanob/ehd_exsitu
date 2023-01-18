# -*- coding: utf-8 -*-
"""
Created on June 15 2022

@author: Oliver Nakano-Baker
"""
# import pickle
import dill

import numpy as np
import pandas as pd

from .mle_models import MLE_Regressor, MLE_Classifier
from .cold_models import (RF_Regressor, MLP_Regressor,
                          RF_Classifier, MLP_Classifier)
from .brute_pretrained_models import (RF_Regressor_Allpre,
                                      RF_Classifier_Allpre,
                                      MLP_Regressor_Allpre,
                                      MLP_Classifier_Allpre,
                                      Ridge_Regressor_Allpre)
                                      # RF_Regressor_lastXY,
                                      # RF_Classifier_lastXY,
                                      # MLP_Regressor_lastXY,
                                      # MLP_Classifier_lastXY)

from .utils import random_N_data_split, dict_mean


def make_model_like(architecture, params):
    model_type = get_model_type(architecture)
    return model_type(params)


def get_model_type(architecture):
    # Regressors
    if architecture == 'MLE'\
       or (architecture == 'normed_MLE'):
        return MLE_Regressor
    if architecture == 'cold_RF':
        return RF_Regressor
    if architecture == 'cold_MLP':
        return MLP_Regressor
    if (architecture == 'only_pretrained_RF')\
       or (architecture == 'normed_RF')\
       or (architecture == 'v_normed_RF'):
        return RF_Regressor_Allpre
    if architecture == 'only_pretrained_MLP'\
       or (architecture == 'v_normed_MLP'):
        return MLP_Regressor_Allpre
    if architecture == 'v_normed_Ridge':
        return Ridge_Regressor_Allpre

    # Classifiers
    if architecture == 'MLE_class'\
       or (architecture == 'normed_MLE_class')\
       or (architecture == 'v_normed_MLE_class'):
        return MLE_Classifier
    if architecture == 'cold_RF_class':
        return RF_Classifier
    if architecture == 'cold_MLP_class':
        return MLP_Classifier
    if architecture == 'only_pretrained_RF_class'\
       or (architecture == 'normed_RF_class')\
       or (architecture == 'v_normed_RF_class'):
        return RF_Classifier_Allpre
    if architecture == 'only_pretrained_MLP_class'\
       or (architecture == 'v_normed_MLP_class'):
        return MLP_Classifier_Allpre

    raise ValueError(f"Not a valid model architecture: {architecture}")


class EHD_Model:
    log_slices = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000,
                           5000, 10_000])
    def __init__(self, architecture, params={}):
        self.architecture = architecture
        self.model = make_model_like(architecture, params)
        self.pretrainer = self.model.pretrainer

    @staticmethod
    def optuna_init(architecture, params, trial):
        model_type = get_model_type(architecture)
        model_params = model_type.optuna_suggest(trial)
        return EHD_Model(architecture,
                         params={**params, **model_params})

    # Some passthrough functions:
    def predict(self, X):
        return self.model.predict(X)

    def pretrain(self, dataset):
        self.model.pretrain(dataset)

    def retrain(self, dataset):
        self.model.retrain(dataset)

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

    def save(self, path):
        with open(path, 'wb') as f:
            dill.dump((self.architecture,
                         self.model.params,
                         self.model),
                         # self.model.pickle()),
                        f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            architecture, params, pick = dill.load(f)
        model = EHD_Model(architecture, params)
        model.model = pick
        # model.model.from_pickle(pick)
        return model
