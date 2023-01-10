# -*- coding: utf-8 -*-
"""
Class of models that uses runtime-measured voltage and pulsewidth thresholds to
normalize the inputs to the model. No extra re-training occurs at run time.
"""
import numpy as np
from copy import deepcopy

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.base import TransformerMixin

from .utils import regression_metrics, classification_metrics


class Normed_Input_Model:
    pretrainer = True
    def __init__(self, params):
        self.params = params
        self.model = None

    def pretrain(self, data):
        pass                    # TODO

    def retrain(self, data):
        pass                    # TODO

    def predict(self, X):
        return self.model.predict(X)


class RF_Regressor_NI(Normed_Input_Model):
    hyperparams = ['n_estimators']

    def __init__(self, params):
        super().__init__(params)
        self.
        self.model = Pipeline([
            ('norm', )
            ('whiten', StandardScaler()),
            ('forest', RandomForestRegressor())
        ])

    def copy(self):
        mycopy = RF_Regressor_NI(self.params.copy())
        mycopy.model = deepcopy(self.model)
        return mycopy

    def tune(self, data):
        pass                    # TODO

    def evaluate(self, data):
        return regression_metrics(data['Y'], self.predict(data['X']))

