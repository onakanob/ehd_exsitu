# -*- coding: utf-8 -*-
"""
Created 10-January-2023

Utility to tune hyperparameters for any model with a .tune() method.

@author: Oliver Nakano-Baker
"""


def tune_hyperparameters(architecture, params, xtype, ytype, filters,
                         N=50):
    """
    Evaluate models with different hyperparameters, report results and
    optimal parameters, and save all trained models for later use as an
    ensemble.

    Architecture is a string describing a model architecture that can be
    interpreted by the function make_model_like() to instate a model.

    Params is a set of keys and parameter _ranges_, i.e. tuples containing
    limits (upper, lower), that represent the hyperparameter search space.

    N is the number of "retraining" points to use for evaluation
    purposes. Lower = fewer retraining points available
    """

    TRIALS = 20

    for T in range(TRIALS):
