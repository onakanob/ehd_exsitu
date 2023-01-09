# -*- coding: utf-8 -*-
"""
Created on June 15 2022

@author: Oliver Nakano-Baker
"""
import numpy as np

from .utils import regression_metrics, classification_metrics


class MLE_Regressor():
    pretrainer = True
    # xtype = 'wave'
    # ytype = 'area'
    # filters = []

    def __init__(self, params):
        self.params = params
        self.Y = 0
        self.observations = 0

    def copy(self):
        mycopy = MLE_Regressor(self.params.copy())
        mycopy.Y = self.Y
        mycopy.observations = self.observations
        return mycopy

    def pretrain(self, data):
        Y = data['Y']
        self.Y = np.mean(Y)
        self.observations += len(Y)

    def retrain(self, data):
        Y = data['Y']
        new_obs = self.observations + len(Y)
        self.Y = (self.Y * self.observations + np.mean(Y) * len(Y)) / new_obs
        self.observations = new_obs

    def predict(self, X):
        return np.array([self.Y for _ in range(X.shape[0])])

    def evaluate(self, data):
        return regression_metrics(data['Y'], self.predict(data['X']))


class MLE_Classifier(MLE_Regressor):
    # xtype = 'wave'
    # ytype = 'jetted_selectors'
    # filters = []

    def __init__(self, params):
        super().__init__(params)
        self.num_classes = None

    def copy(self):
        mycopy = MLE_Classifier(self.params.copy())
        mycopy.Y = self.Y
        mycopy.observations = self.observations
        mycopy.num_classes = self.num_classes
        return mycopy

    def update_Y(self):
        self.Y = np.argmax(self.observations)
        # self.Y = np.zeros(self.num_classes)
        # self.Y[np.argmax(self.observations)] = 1

    def pretrain(self, data):
        Y = idx_to_onehot(data['Y'])
        self.num_classes = Y.shape[1]
        self.observations = Y.sum(0)
        self.update_Y()

    def retrain(self, data):
        Y = idx_to_onehot(data['Y'])
        self.observations += Y.sum(0)
        self.update_Y()

    def evaluate(self, data):
        return classification_metrics(data['Y'], self.predict(data['X']))


def idx_to_onehot(idxs, num_classes=None):
    if num_classes is None:
        num_classes = np.max(idxs + 1)
    b = np.zeros((len(idxs), num_classes))
    b[np.arange(len(idxs)), idxs.astype(int)] = 1
    return b
