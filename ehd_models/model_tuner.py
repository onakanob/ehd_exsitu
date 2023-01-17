# -*- coding: utf-8 -*-
"""
Created 10-January-2023

Utility to tune hyperparameters for any model with a .tune() method.

@author: Oliver Nakano-Baker
"""
import gc
import os
import time

import numpy as np
import optuna

from . import EHD_Model


def train_model(trial, config):
    """Instate, train, and evaluate a model based on values in the config
    dictionary. Assume we receive a dataset loader object in the
    config['loader']"""
    pretrain_set, eval_set, eval_name = config['dataset']
    trial.set_user_attr('eval_dataset', eval_name)

    model = EHD_Model.optuna_init(architecture=config['architecture'],
                                  params=config,
                                  trial=trial)
    model.pretrain(pretrain_set)

    output = model.evaluate(eval_set, train_sizes=[config['N']])
    for key, value in output.items():
        trial.set_user_attr(key, float(value))

    outfile = os.path.join(config['output_dir'],
                           f"{eval_name}-model_{trial._trial_id}.pickle")
    model.save(outfile)

    return float(output['F1']) * float(output['AUC'])\
        * float(output['class 1 precision'])


def tune_hyperparameters(architecture, xtype, ytype, filters, loader,
                         fold=0, trials=3, time_limit=1, output_dir='.',
                         max_concurrent=1, N=50):
    """
    Evaluate models with different hyperparameters, report results and
    optimal parameters, and save all trained models for later use as an
    ensemble.

    Architecture is a string describing a model architecture that can be
    interpreted by the function make_model_like() to instate a model.

    Params is a set of keys and parameter _ranges_, i.e. tuples containing
    limits (name, upper, lower), that represent the hyperparameter search space.

    N is the number of "retraining" points to use for evaluation
    purposes. Lower = fewer retraining points available
    """
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    config = {'architecture': architecture,
              'N': N,
              'output_dir': os.path.abspath(output_dir),
              'dataset': loader.folded_dataset(fold=fold,
                                               xtype=xtype,
                                               ytype=ytype,
                                               pretrain=True,
                                               filters=filters)
              }

    study = optuna.create_study(direction='maximize')
    objective = lambda trial: train_model(trial, config)
    study.optimize(objective,
                   n_trials=trials,
                   n_jobs=max_concurrent,
                   timeout=time_limit*60)
    study.trials_dataframe().to_excel(os.path.join(output_dir, "results.xlsx"))
