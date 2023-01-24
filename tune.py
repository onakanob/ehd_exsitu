# -*- coding: utf-8 -*-
"""
Created 12-January-2023

Tune hyperparameters for any model with a .tune() method.

@author: Oliver Nakano-Baker
"""
import os
import argparse

import numpy as np
# from ray import tune

from ehd_dataset import EHD_Loader
from ehd_models.model_tuner import tune_hyperparameters

# Experiment: Define the model type and dataset parameters
ARCHITECTURE = "scaling_MLP"     # normed_MLP_class
XTYPE = "normed_squares"     # vector, normed_squares, v_normed_squares
YTYPE = "max_width"          # jetted, max_width
FILTERS = [                  # Std tip square waves with threshold measured
           ('vector', lambda x: len(x), 2),
           ('Wavegen', lambda x: x, 'square'),  # harmonics, square
           ('V Thresh [V] @ .5s', np.isnan, False),
           ('SIJ Tip', lambda x: x, 'Std'),
           ('clogging', lambda x: x, False),
           ('jetted',  lambda x: x, True),  # For regression only
          ]


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default="./TUNE_OUTPUT",
                        help='''Directory to output results and save
                        models.''')
    parser.add_argument('--index', type=str,
        default="C:/Dropbox/SPEED/Self Driving EHD/Datasets/dataset_index.xlsx",
                        help='''Path to dataset index excel file.''')
    parser.add_argument('--minutes', type=int, default=1,
                        help='''Minutes to run.''')
    parser.add_argument('--trials', type=int, default=100,
                        help='''Trials to run.''')
    parser.add_argument('--parallel', type=int, default=1,
                        help='''Max concurrent trials to run in parallel.''')
    args = parser.parse_args()

    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    ehd_loader = EHD_Loader(args.index)

    summaries = []
    for fold in range(ehd_loader.num_folds(FILTERS)):
        output_dir = os.path.join(args.output,
                                  f'fold_{fold}')
        summary = tune_hyperparameters(ARCHITECTURE, XTYPE, YTYPE, FILTERS,
                                       loader=ehd_loader,
                                       fold=fold,
                                       trials=args.trials,
                                       time_limit=args.minutes,
                                       max_concurrent=args.parallel,
                                       output_dir=output_dir)
        summaries.append(summary)

    # Print summary
    cols = summaries[0].columns
    best_idx = [s['value'].argmax() for s in summaries]
    if 'user_attrs_eval_dataset' in cols:
        print('dataset:')
        print([s['user_attrs_eval_dataset'].loc[best_idx[i]] for i, s in enumerate(summaries)])
    if 'value' in cols:
        print('values:')
        print([s['value'].loc[best_idx[i]] for i, s in enumerate(summaries)])
    if 'user_attrs_MSE' in cols:
        print('MSEs:')
        print([s['user_attrs_MSE'].loc[best_idx[i]] for i, s in enumerate(summaries)])
    if 'user_attrs_r' in cols:
        print('r:')
        print([s['user_attrs_r'].loc[best_idx[i]] for i, s in enumerate(summaries)])
