# -*- coding: utf-8 -*-
"""
Methods for EHD machine learning where training only happens during the
'pretraining' step.

Created on July 26 2022

@author: Oliver Nakano-Baker
"""
from copy import deepcopy

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from .brute_pretrained_models import (MLP_Regressor_Allpre,
                                      RF_Regressor_Allpre)


class MLP_Regressor_Warmstart(MLP_Regressor_Allpre):
    def __init__(self, params):
        super().__init__(params)
        self.pipe['MLP'].set_params(warm_start=True)

    @staticmethod
    def optuna_suggest(trial):
        d = MLP_Regressor_Allpre.optuna_suggest(trial)
        d['retrain_lr'] = trial.suggest_float('retrain_lr', 1e-5, 1e-2,
                                              log=True)
        d['retrain_alpha'] = trial.suggest_float('retrain_alpha', 1e-6, 1e-3,
                                                 log=True)
        return d

    def retrain(self, data):
        if len(data['Y']) > 10:
            self.pipe['MLP'].set_params(warm_start=True)
            self.pipe['MLP'].set_params(batch_size='auto')
            if self.params.get('retrain_lr') is not None:
                self.pipe['MLP'].set_params(learning_rate_init=self.params['retrain_lr'])
            if self.params.get('retrain_alpha') is not None:
                self.pipe['MLP'].set_params(alpha=self.params['retrain_alpha'])
            self.pipe.fit(data['X'], data['Y'])

    def copy(self):
        mycopy = MLP_Regressor_Warmstart(self.params.copy())
        mycopy.pipe = deepcopy(self.pipe)
        return mycopy


class MLP_Regressor_Scaling(MLP_Regressor_Allpre):
    def __init__(self, params):
        super().__init__(params)
        final_alpha = self.params.get('final_alpha')
        if final_alpha is None:
            final_alpha = 1.0
        self.final_estimator = Ridge(alpha=final_alpha)

    @staticmethod
    def optuna_suggest(trial):
        d = MLP_Regressor_Allpre.optuna_suggest(trial)
        d['final_alpha'] = trial.suggest_float('final_alpha', 1e-3, 1e1)
        return d

    def retrain(self, data):
        X = self.pipe.predict(data['X'])
        self.final_estimator.fit(X, data['Y'])

    def predict(self, X):
        X = self.pipe.predict(X)
        return self.final_estimator.predict(X)


class RF_Regressor_Warmstart(RF_Regressor_Allpre):
    def __init__(self, params):
        super().__init__(params)
        
    def retrain(self, data):
        new_trees = self.params.get('new_trees')
        if new_trees is None:
            new_trees = 100

        self.pipe['forest'].set_params(warm_start=True)
        self.pipe['forest'].set_params(
            n_estimators=int(self.pipe['forest'].n_estimators
                             + new_trees)
        )
        self.pipe.fit(data['X'], data['Y'])

    @staticmethod
    def optuna_suggest(trial):
        d = RF_Regressor_Allpre.optuna_suggest(trial)
        d['new_trees'] = trial.suggest_int('new_trees', 10, 200)
        return d

    def copy(self):
        mycopy = RF_Regressor_Warmstart(self.params.copy())
        mycopy.pipe = deepcopy(self.pipe)
        return mycopy


class RF_Regressor_Reweight(RF_Regressor_Allpre):
    def __init__(self, params):
        super().__init__(params)
        alpha = self.params.get('alpha')
        if alpha is None:
            alpha = 1.0
        self.final_estimator = Pipeline([
            ('whiten', StandardScaler()),
            ('Ridge', Ridge(alpha=alpha))
        ])

    def retrain(self, data):
        X = self.pipe['whiten'].transform(data['X'])
        X = np.array([m.predict(X) for m in
                                self.pipe['forest'].estimators_]).T
        self.final_estimator.fit(X, data['Y'].astype(float))

    def predict(self, X):
        X = self.pipe['whiten'].transform(X)
        X = np.array([m.predict(X) for m in self.pipe['forest'].estimators_]).T
        return self.final_estimator.predict(X)
        

    @staticmethod
    def optuna_suggest(trial):
        d = RF_Regressor_Allpre.optuna_suggest(trial)
        d['alpha'] = trial.suggest_float('alpha', 1e-5, 1e-1)
        return d
