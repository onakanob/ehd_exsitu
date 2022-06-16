# -*- coding: utf-8 -*-
"""
Created on June 15 2022

@author: Oliver Nakano-Baker
"""
from .mle_models import MLE_Regressor, MLE_Classifier


def switch_architectures(architecture, params):
    if architecture == 'MLE':
        return MLE_Regressor(params)
    if architecture == 'MLE_class':
        return MLE_Classifier(params)


class EHD_Model:
    def __init__(self, architecture, params={}):
        self.architecture = architecture
        self.model = switch_architectures(architecture, params)

    def pretrain(self, dataset):
        pass

    def evaluate(self, dataset, train_sizes=None):
        return {'train_size': 0, 'AUC': 0}
