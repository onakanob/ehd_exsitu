# -*- coding: utf-8 -*-
"""
Random Forest methods for EHD machine learning.

Created on June 23 2022

@author: Oliver Nakano-Baker
"""
import pickle

import numpy as np
from copy import deepcopy

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier

from .utils import regression_metrics, classification_metrics


class Cold_SciKit_Model:
    pretrainer = False
    def __init__(self, params):
        self.params = params

    def pretrain(self, data):
        """A cold model has no pretraining behavior."""
        pass

    def retrain(self, data):
        self.pipe.fit(data['X'], data['Y'])

    def predict(self, X):
        return self.pipe.predict(X)

    def pickle(self):
        return pickle.dumps(self.pipe)

    def from_pickle(self, pick):
        self.pipe = pickle.loads(pick)



class RF_Regressor(Cold_SciKit_Model):
    hyperparams = ['n_estimators', 'min_samples_split']  # TODO etc

    # Dataset specifiers - will be passed to EHD_Loader.folded_dataset
    # xtype = 'vector'
    # ytype = 'area'
    # filters = [('vector', lambda x: len(x), 6)]
    # filters = []

    def __init__(self, params):
        super().__init__(params)
        # TODO introduce hyperparams - defaults for now
        # self.pipe = RandomForestRegressor()
        self.pipe = Pipeline([
            ('whiten', StandardScaler()),
            ('forest', RandomForestRegressor())
        ])

    def copy(self):
        mycopy = RF_Regressor(self.params.copy())
        mycopy.pipe = deepcopy(self.pipe)
        return mycopy

    def tune(self, data):
        pass                    # TODO

    def evaluate(self, data):
        return regression_metrics(data['Y'], self.predict(data['X']))


class MLP_Regressor(Cold_SciKit_Model):
    # xtype = 'vector'
    # ytype = 'area'
    # filters = [('vector', lambda x: len(x), 6)]
    # filters = []

    def __init__(self, params):
        super().__init__(params)
        # TODO introduce hyperparams - defaults for now
        self.pipe = Pipeline([
            ('whiten', StandardScaler()),
            ('forestMLP', MLPRegressor(max_iter=params['max_iter']))
        ])

    def copy(self):
        mycopy = MLP_Regressor(self.params.copy())
        mycopy.pipe = deepcopy(self.pipe)
        return mycopy

    def tune(self, data):
        pass                    # TODO

    def evaluate(self, data):
        return regression_metrics(data['Y'], self.predict(data['X']))    


class RF_Classifier(Cold_SciKit_Model):
    # xtype = 'vector'
    # ytype = 'jetted'
    # filters = [('vector', lambda x: len(x), 6)]
    # filters = []

    def __init__(self, params):
        super().__init__(params)
        # TODO introduce hyperparams
        self.pipe = Pipeline([
            ('whiten', StandardScaler()),
            ('forest', RandomForestClassifier())
        ])

    def copy(self):
        mycopy = RF_Classifier(self.params.copy())
        mycopy.pipe = deepcopy(self.pipe)
        return mycopy

    def tune(self, data):
        pass                    # TODO

    def evaluate(self, data):
        # import ipdb; ipdb.set_trace()
        return classification_metrics(data['Y'], self.predict(data['X']))


class MLP_Classifier(Cold_SciKit_Model):
    # xtype = 'vector'
    # ytype = 'jetted'
    # filters = [('vector', lambda x: len(x), 6)]
    # filters = []

    def __init__(self, params):
        super().__init__(params)
        # TODO introduce hyperparams
        self.pipe = Pipeline([
            ('whiten', StandardScaler()),
            ('MLP', MLPClassifier(max_iter=params['max_iter']))
        ])

    def copy(self):
        mycopy = MLP_Classifier(self.params.copy())
        mycopy.pipe = deepcopy(self.pipe)
        return mycopy

    def tune(self, data):
        pass                    # TODO

    def evaluate(self, data):
        return classification_metrics(data['Y'], self.predict(data['X']))
