# -*- coding: utf-8 -*-
"""
Random Forest methods for EHD machine learning.

Created on June 23 2022

@author: Oliver Nakano-Baker
"""
import numpy as np
from copy import deepcopy

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier

from .utils import regression_metrics, classification_metrics


class Cold_SciKit_Model:
    def __init__(self, params):
        self.params = params

    def pretrain(self, data):
        """A cold model has no pretraining behavior."""
        pass

    def retrain(self, data):
        self.model.fit(data['X'], data['Y'])

    def predict(self, X):
        return self.model.predict(X)


class RF_Regressor(Cold_SciKit_Model):
    hyperparams = ['n_estimators', 'min_samples_split']  # TODO etc

    # Dataset specifiers - will be passed to EHD_Loader.folded_dataset
    xtype = 'vector'
    ytype = 'area'
    filters = [('vector', lambda x: len(x), 6)]
    
    def __init__(self, params):
        super().__init__(params)
        # TODO introduce hyperparams - defaults for now
        # self.model = RandomForestRegressor()
        self.model = Pipeline([
            ('whiten', StandardScaler()),
            ('forest', RandomForestRegressor())
        ])

    def copy(self):
        mycopy = RF_Regressor(self.params.copy())
        mycopy.model = deepcopy(self.model)
        return mycopy

    def tune(self, data):
        pass                    # TODO

    def evaluate(self, data):
        return regression_metrics(data['Y'], self.predict(data['X']))


class MLP_Regressor(Cold_SciKit_Model):
    xtype = 'vector'
    ytype = 'area'
    filters = [('vector', lambda x: len(x), 6)]

    def __init__(self, params):
        super().__init__(params)
        # TODO introduce hyperparams - defaults for now
        self.model = Pipeline([
            ('whiten', StandardScaler()),
            ('forest', MLPRegressor(max_iter=3000))
        ])

    def copy(self):
        mycopy = MLP_Regressor(self.params.copy())
        mycopy.model = deepcopy(self.model)
        return mycopy

    def tune(self, data):
        pass                    # TODO

    def evaluate(self, data):
        return regression_metrics(data['Y'], self.predict(data['X']))    


class RF_Classifier(Cold_SciKit_Model):
    xtype = 'vector'
    ytype = 'jetted'
    filters = [('vector', lambda x: len(x), 6)]

    def __init__(self, params):
        super().__init__(params)
        # TODO introduce hyperparams
        self.model = Pipeline([
            ('whiten', StandardScaler()),
            ('forest', RandomForestClassifier())
        ])

    def copy(self):
        mycopy = RF_Classifier(self.params.copy())
        mycopy.model = deepcopy(self.model)
        return mycopy

    def tune(self, data):
        pass                    # TODO

    def evaluate(self, data):
        return classification_metrics(data['Y'], self.predict(data['X']))


class MLP_Classifier(Cold_SciKit_Model):
    xtype = 'vector'
    ytype = 'jetted'
    filters = [('vector', lambda x: len(x), 6)]

    def __init__(self, params):
        super().__init__(params)
        # TODO introduce hyperparams
        self.model = Pipeline([
            ('whiten', StandardScaler()),
            ('forest', MLPClassifier(max_iter=10000))
        ])

    def copy(self):
        mycopy = MLP_Classifier(self.params.copy())
        mycopy.model = deepcopy(self.model)
        return mycopy

    def tune(self, data):
        pass                    # TODO

    def evaluate(self, data):
        return classification_metrics(data['Y'], self.predict(data['X']))
