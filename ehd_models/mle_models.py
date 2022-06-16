# -*- coding: utf-8 -*-
"""
Created on June 15 2022

@author: Oliver Nakano-Baker
"""
import numpy as np


class MLE_Regressor():
    def __init__(self, params):
        self.params = params
        self.Y = 0
        self.observations = 0

    def pretrain(self, X, Y):
        self.Y = np.mean(Y)
        self.observations += len(Y)

    def retrain(self, X, Y):
        new_obs = self.observations + len(Y)
        self.Y = (self.Y * self.observations + np.mean(Y) * len(Y)) / new_obs
        self.observations = new_obs

    def predict(self, X):
        return np.array([self.Y for _ in range(X.shape[0])])


class MLE_Classifier():
    def __init__(self, params):
        self.params = params
        self.Y = None
        self.num_classes = None
        self.observations = None

    def pretrain(self, X, Y):
        self.num_classes = Y.shape[1]
        self.observations = Y.sum(0)
        self.Y = np.zeros(num_classes)
        self.Y[np.argmax(self.observations)] = 1

    def retrain(self, X, Y):
        self.observations += Y.sum(0)
        self.Y = np.zeros(num_classes)
        self.Y[np.argmax(self.observations)] = 1

    def predict(self, X):
        return np.array([self.Y for _ in range(X.shape[0])])
