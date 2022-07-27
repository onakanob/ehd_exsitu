# -*- coding: utf-8 -*-
"""
Methods for EHD machine learning where training only happens during the
'pretraining' step.

Created on July 26 2022

@author: Oliver Nakano-Baker
"""
import numpy as np
from copy import deepcopy

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from .utils import regression_metrics, classification_metrics


class Only_Hot_SciKit_Model:
    pretrainer = True
    def __init__(self, params):
        self.params = params

    def pretrain(self, data):
        self.model.fit(data['X'], data['Y'])

    def retrain(self, data):
        """A hot model has no retraining behavior."""
        pass

    def predict(self, X):
        return self.model.predict(X)


class RF_Regressor_Allpre(Only_Hot_SciKit_Model):
    hyperparams = ['n_estimators', 'min_samples_split']  # TODO etc

    # Dataset specifiers - will be passed to EHD_Loader.folded_dataset
    xtype = 'vector'
    ytype = 'area'
    filters = [('vector', lambda x: len(x), 6)]

    def __init__(self, params):
        super().__init__(params)
        self.model = Pipeline([
            ('whiten', StandardScaler()),
            ('forest', RandomForestRegressor())
        ])

    def copy(self):
        mycopy = RF_Regressor_Allpre(self.params.copy())
        mycopy.model = deepcopy(self.model)
        return mycopy

    def tune(self, data):
        pass                    # TODO

    def evaluate(self, data):
        return regression_metrics(data['Y'], self.predict(data['X']))


class RF_Regressor_lastXY(RF_Regressor_Allpre):
    def __init__(self, params):
        super().__init__(params)
        self.xtype = 'last_vector'

    def copy(self):
        mycopy = RF_Regressor_lastXY(self.params.copy())
        mycopy.model = deepcopy(self.model)
        return mycopy


class RF_Classifier_Allpre(Only_Hot_SciKit_Model):
    hyperparams = ['n_estimators', 'min_samples_split']  # TODO etc
    xtype = 'vector'
    ytype = 'jetted'
    filters = [('vector', lambda x: len(x), 6)]

    def __init__(self, params):
        super().__init__(params)
        self.model = Pipeline([
            ('whiten', StandardScaler()),
            ('forest', RandomForestClassifier())
        ])

    def copy(self):
        mycopy = RF_Classifier_Allpre(self.params.copy())
        mycopy.model = deepcopy(self.model)
        return mycopy

    def tune(self, data):
        pass                    # TODO

    def evaluate(self, data):
        return classification_metrics(data['Y'], self.predict(data['X']))


class RF_Classifier_lastXY(RF_Classifier_Allpre):
    def __init__(self, params):
        super().__init__(params)
        self.xtype = 'last_vector'

    def copy(self):
        mycopy = RF_Classifier_lastXY(self.params.copy())
        mycopy.model = deepcopy(self.model)
        return mycopy
