# -*- coding: utf-8 -*-
"""
Methods for EHD machine learning where training only happens during the
'pretraining' step.

Created on July 26 2022

@author: Oliver Nakano-Baker
"""
import pickle

import numpy as np
from copy import deepcopy

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.linear_model import Ridge

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

    def pickle(self):
        return pickle.dumps(self.pipe)

    def from_pickle(self, pick):
        self.pipe = pickle.loads(pick)


class Only_Hot_Classifier(Only_Hot_SciKit_Model):
    def evaluate(self, data):
        return classification_metrics(data['Y'], self.predict(data['X']))


class Only_Hot_Regressor(Only_Hot_SciKit_Model):
    def evaluate(self, data):
        return regression_metrics(data['Y'], self.predict(data['X']))


# >>>>>>>> Ridge Regression Model <<<<<<<<
class Ridge_Model():
    PARAMS = ('alpha', 'fit_intercept', 'solver')

    @staticmethod
    def optuna_suggest(trial):
        return {
            'alpha':         trial.suggest_float('alpha', 1e-3, 1e2,
                                                 log=True),
            'fit_intercept': trial.suggest_categorical('fit_intercept',
                                                       [True, False]),
            'solver':        trial.suggest_categorical('solver',
                                                       ['svd', 'cholesky',
                                                        'lsqr', 'sag',
                                                        'lbfgs']),
        }

    def __init__(self, params):
        self.ridge_parameters = {}
        for key in self.PARAMS:
            value = params.get(key)
            if value is not None:
                self.ridge_parameters[key] = value
        if self.ridge_parameters.get('solver') is not None\
           and self.ridge_parameters.get('solver') == 'lbfgs':
            self.ridge_parameters['positive'] = True
        # inheritor must create the pipeline

class Ridge_Regressor_Allpre(Only_Hot_Regressor, Ridge_Model):
    def __init__(self, params):
        super().__init__(params)
        Ridge_Model.__init__(self, params)

        self.pipe = Pipeline([
            ('whiten', StandardScaler()),
            ('Ridge', Ridge(**self.ridge_parameters))
        ])

    def copy(self):
        mycopy = Ridge_Regressor_Allpre(self.params.copy())
        mycopy.pipe = deepcopy(self.pipe)
        return mycopy


# >>>>>>>> Random Forest Models <<<<<<<<
class RF_Model():
    PARAMS = ('n_estimators', 'max_depth', 'min_samples_split',
              'min_samples_leaf', 'max_features', 'max_leaf_nodes',
              'bootstrap', 'max_samples')

    @staticmethod
    def optuna_suggest(trial):
        return {
            'n_estimators':      trial.suggest_int('n_estimators', 10, 1000),
            'max_depth':         trial.suggest_int('max_depth', 1, 100),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
            'min_samples_leaf':  trial.suggest_int('min_samples_leaf', 1, 4),
            'max_leaf_nodes':    trial.suggest_int('max_leaf_nodes', 2, 100),
            'bootstrap':         trial.suggest_categorical('bootstrap', [True, False]),
            'max_samples':       trial.suggest_int('max_samples', 10, 50),
        }

    def __init__(self, params):
        self.forest_parameters = {}
        for key in self.PARAMS:
            value = params.get(key)
            if value is not None:
                self.forest_parameters[key] = value
        if self.forest_parameters.get("bootstrap") is not None\
           and not self.forest_parameters.get("bootstrap"):
            self.forest_parameters["max_samples"] = None
        # inheritor must create the pipeline


class RF_Regressor_Allpre(Only_Hot_Regressor, RF_Model):
    # hyperparams = ['n_estimators', 'min_samples_split']  # TODO etc

    def __init__(self, params):
        super().__init__(params)
        RF_Model.__init__(self, params)

        self.pipe = Pipeline([
            ('whiten', StandardScaler()),
            ('forest', RandomForestRegressor(**self.forest_parameters))
        ])

    def copy(self):
        mycopy = RF_Regressor_Allpre(self.params.copy())
        mycopy.pipe = deepcopy(self.pipe)
        return mycopy


class RF_Classifier_Allpre(Only_Hot_Classifier, RF_Model):
    def __init__(self, params):
        super().__init__(params)
        RF_Model.__init__(self, params)

        self.pipe = Pipeline([
            ('whiten', StandardScaler()),
            ('forest', RandomForestClassifier(**self.forest_parameters))
        ])

    def copy(self):
        mycopy = RF_Classifier_Allpre(self.params.copy())
        mycopy.pipe = deepcopy(self.pipe)
        return mycopy


# >>>>>>>> 2-layer Perceptron Models <<<<<<<<
class MLP_Model():
    PARAMS = ['activation', 'alpha', 'batch_size', 'learning_rate_init',
              'beta_1', 'beta_2', 'epsilon', 'max_iter']

    def __init__(self, params):
        self.mlp_params = {
            'early_stopping': True,
            'max_iter': 3000,
            'solver': 'adam',
            'learning_rate': 'adaptive',
            'epsilon': 1e-8,
        }
        if params.get('layer1_size') and params.get('layer2_size'):
            self.mlp_params['hidden_layer_sizes'] = [params['layer1_size'],
                                                     params['layer2_size']]
        for key in self.PARAMS:
            value = params.get(key)
            if value is not None:
                self.mlp_params[key] = value
        if self.mlp_params.get('beta_1') is not None:
            self.mlp_params['beta_1'] = 1 - self.mlp_params['beta_1']
        if self.mlp_params.get('beta_2') is not None:
            self.mlp_params['beta_2'] = 1 - self.mlp_params['beta_2']

        # inheritor must create the pipeline

    @staticmethod
    def optuna_suggest(trial):
        return {
            'activation':         trial.suggest_categorical('activation',
                                                            ['tanh', 'relu']),
            'batch_size':         trial.suggest_int('batch_size', 20, 200,
                                                    log=False),
            'layer1_size':        trial.suggest_int('layer1_size', 20, 2000,
                                                    log=True),
            'layer2_size':        trial.suggest_int('layer2_size', 20, 2000,
                                                    log=True),
            'learning_rate_init': trial.suggest_float('lr', 1e-5, 1e-2,
                                                      log=True),
            'alpha':              trial.suggest_float('alpha', 1e-6, 1e-3,
                                                      log=True),
            'beta_1':             trial.suggest_float('beta_1', .05, .2,
                                                      log=True),
            'beta_2':             trial.suggest_float('beta_2', .0005, .002,
                                                      log=True),
        }


class MLP_Regressor_Allpre(Only_Hot_Regressor, MLP_Model):
    def __init__(self, params):
        super().__init__(params)
        MLP_Model.__init__(self, params)

        self.pipe = Pipeline([
            ('whiten', StandardScaler()),
            ('MLP', MLPRegressor(**self.mlp_params))
        ])

    def copy(self):
        mycopy = MLP_Regressor_Allpre(self.params.copy())
        mycopy.pipe = deepcopy(self.pipe)
        return mycopy


class MLP_Classifier_Allpre(Only_Hot_Classifier, MLP_Model):
    def __init__(self, params):
        super().__init__(params)
        MLP_Model.__init__(self, params)

        self.pipe = Pipeline([
            ('whiten', StandardScaler()),
            ('MLP', MLPClassifier(**self.mlp_params))
        ])

    def copy(self):
        mycopy = MLP_Classifier_Allpre(self.params.copy())
        mycopy.pipe = deepcopy(self.pipe)
        return mycopy
