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


def dict_cat(dicts):
    """Concatenate each value in an iterable of dictionaries."""
    keys = np.unique([k for d in dicts for k in d.keys()])
    catted = None
    for d in dicts:
        if catted is None:
            catted = d
        else:
            for k in keys:
                catted[k] = np.concatenate([catted[k], d[k]])
    return catted


def train_model(trial, config, validation_count=2):
    """Instate, train, and evaluate a model based on values in the config
    dictionary. Assume we receive a dataset loader object in the
    config['loader']"""
    # pretrain_set, eval_set, eval_name = config['dataset']
    datasets = config['datasets']
    names = config['dataset_names']
    arange = np.arange(len(names))
    eval_idx = np.sort(np.random.choice(arange,
                                        size=validation_count,
                                        replace=False))
    eval_set = dict_cat(datasets[eval_idx])
    pretrain_set = dict_cat(datasets[[n not in eval_idx for n in arange]])
    
    trial.set_user_attr('eval_datasets', str(names[eval_idx]))

    model = EHD_Model.optuna_init(architecture=config['architecture'],
                                  params=config,
                                  trial=trial)
    model.pretrain(pretrain_set)

    output = model.evaluate(eval_set, train_sizes=[config['N']])
    for key, value in output.items():
        trial.set_user_attr(key, float(value))

    outfile = os.path.join(config['output_dir'],
                           f"model_{trial._trial_id}.pickle")
    model.save(outfile)

    # What to maximize:
    if 'F1' in output.columns:  # Classification
        return float(output['F1']) * float(output['AUC'])\
            * float(output['class 1 precision'])
    elif 'MSE' in output.columns:  # Regression
        return -float(output['MSE'])
    else:
        raise ValueError('Unrecognized model metrics')


def tune_hyperparameters(architecture, xtype, ytype, filters, loader,
                         trials=3, time_limit=1, output_dir='.',
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

    # Assemble a list of datasets
    datasets = []
    names = []
    for fold in range(loader.num_folds(filters)):  # Ugly as sin:
        _, dataset, name = loader.folded_dataset(
            fold=fold,
            xtype=xtype,
            ytype=ytype,
            pretrain=False,
            filters=filters
        )
        datasets.append(dataset)
        names.append(name)

    config = {'architecture': architecture,
              'N': N,
              'output_dir': os.path.abspath(output_dir),
              'datasets': np.array(datasets),
              'dataset_names': np.array(names),
              }

    study = optuna.create_study(direction='maximize')
    objective = lambda trial: train_model(trial, config)
    study.optimize(objective,
                   n_trials=trials,
                   n_jobs=max_concurrent,
                   timeout=time_limit*60)

    df = study.trials_dataframe()
    df.to_excel(os.path.join(output_dir, "results.xlsx"))
    return df
