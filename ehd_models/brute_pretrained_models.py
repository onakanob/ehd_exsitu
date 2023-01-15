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
from sklearn.neural_network import MLPRegressor, MLPClassifier

from .utils import regression_metrics, classification_metrics


class Only_Hot_SciKit_Model:
    pretrainer = True
    def __init__(self, params):
        self.params = params

    def pretrain(self, data):
        self.pipe.fit(data['X'], data['Y'])

    def retrain(self, data):
        """A hot model has no retraining behavior."""
        pass

    def predict(self, X):
        return self.pipe.predict(X)


class RF_Regressor_Allpre(Only_Hot_SciKit_Model):
    hyperparams = ['n_estimators', 'min_samples_split']  # TODO etc

    def __init__(self, params):
        super().__init__(params)
        self.pipe = Pipeline([
            ('whiten', StandardScaler()),
            ('forest', RandomForestRegressor())
        ])

    def copy(self):
        mycopy = RF_Regressor_Allpre(self.params.copy())
        mycopy.pipe = deepcopy(self.pipe)
        return mycopy

    def tune(self, data):
        pass                    # TODO

    def evaluate(self, data):
        return regression_metrics(data['Y'], self.predict(data['X']))


# class RF_Regressor_lastXY(RF_Regressor_Allpre):
#     def __init__(self, params):
#         super().__init__(params)
#         self.xtype = 'last_vector'

#     def copy(self):
#         mycopy = RF_Regressor_lastXY(self.params.copy())
#         mycopy.pipe = deepcopy(self.pipe)
#         return mycopy


class RF_Classifier_Allpre(Only_Hot_SciKit_Model):
    PARAMS = ('n_estimators', 'max_depth', 'min_samples_split',
              'min_samples_leaf', 'max_features', 'max_leaf_nodes',
              'bootstrap', 'max_samples')

    def __init__(self, params):
        super().__init__(params)

        forest_parameters = {}
        for key in self.PARAMS:
            value = params.get(key)
            if value is not None:
                forest_parameters[key] = value
        if forest_parameters.get("bootstrap") is not None\
           and not forest_parameters.get("bootstrap"):
            forest_parameters["max_samples"] = None

        self.pipe = Pipeline([
            ('whiten', StandardScaler()),
            ('forest', RandomForestClassifier(**forest_parameters))
        ])

    def copy(self):
        mycopy = RF_Classifier_Allpre(self.params.copy())
        mycopy.pipe = deepcopy(self.pipe)
        return mycopy

    def evaluate(self, data):
        return classification_metrics(data['Y'], self.predict(data['X']))


# class RF_Classifier_lastXY(RF_Classifier_Allpre):
#     def __init__(self, params):
#         super().__init__(params)
#         self.xtype = 'last_vector'

#     def copy(self):
#         mycopy = RF_Classifier_lastXY(self.params.copy())
#         mycopy.pipe = deepcopy(self.pipe)
#         return mycopy


class MLP_Regressor_Allpre(RF_Regressor_Allpre):
    def __init__(self, params):
        super().__init__(params)
        # self.xtype = 'last_vector'
        self.pipe = Pipeline([
            ('whiten', StandardScaler()),
            ('MLP', MLPRegressor(max_iter=params['max_iter']))
        ])

    def copy(self):
        mycopy = MLP_Regressor_lastXY(self.params.copy())
        mycopy.pipe = deepcopy(self.pipe)
        return mycopy


class MLP_Classifier_Allpre(RF_Classifier_Allpre):
    def __init__(self, params):
        super().__init__(params)
        # self.xtype = 'last_vector'
        self.pipe = Pipeline([
            ('whiten', StandardScaler()),
            ('MLP', MLPClassifier(max_iter=params['max_iter']))
        ])

    def copy(self):
        mycopy = MLP_Classifier_lastXY(self.params.copy())
        mycopy.pipe = deepcopy(self.pipe)
        return mycopy
