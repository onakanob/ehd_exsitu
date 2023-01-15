# -*- coding: utf-8 -*-
"""
Created 10-January-2023

Utility to tune hyperparameters for any model with a .tune() method.

@author: Oliver Nakano-Baker
"""
import os

import numpy as np
from ray import tune
import ray

from . import EHD_Model


def train_model(config):
    """Instate, train, and evaluate a model based on values in the config
    dictionary. Assume we receive a dataset loader object in the
    config['loader']"""
    model = EHD_Model(architecture=config['architecture'],
                      params=config)
    loader = config['loader']

    num_folds = loader.num_folds(config['filters'])
    fold = np.random.randint(num_folds)
    pretrain_set, eval_set, eval_name =\
        loader.folded_dataset(fold=fold,
                              xtype=config['xtype'],
                              ytype=config['ytype'],
                              pretrain=model.pretrainer,
                              filters=config['filters'])
    model.pretrain(pretrain_set)
    output = model.evaluate(eval_set, train_sizes=[config['N']])

    # TODO save the model to output/model_{i}.pickle

    return {'eval_dataset': eval_name,
            'F1': float(output['F1']),
            'AUC': float(output['AUC']),
            'Precision': float(output['class 1 precision']),
            'Recall': float(output['class 1 recall']),}


def tune_hyperparameters(architecture, xtype, ytype, filters, params, loader,
                         trials=3, N=50, output_dir='.'):
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

    # Busywork...
    config = {'architecture': architecture,
              'xtype': xtype,
              'ytype': ytype,
              'filters': filters,
              'N': N,
              'output_dir': output_dir,
              'loader': loader,
              'trial': tune.grid_search(range(trials)),
              **params}

    # train function, wrapped for parallelism
    par_train = tune.with_resources(train_model, {"cpu": 1})

    tuner = tune.Tuner(par_train, param_space=config)
    results = tuner.fit()

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    results.get_dataframe().to_excel(os.path.join(output_dir, "results.xlsx"))
