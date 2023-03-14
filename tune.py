# -*- coding: utf-8 -*-
"""
Created 12-January-2023

Tune hyperparameters for any model with a .tune() method.

@author: Oliver Nakano-Baker
"""
import os
import argparse
from shutil import copy2

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
    parser.add_argument('--pickle', type=str,
        default="C:/Dropbox/SPEED/Self Driving EHD/Datasets/compiled_data.pickle",
                        help='''Path to dataset pickle file.''')
    parser.add_argument('--minutes', type=int, default=1,
                        help='''Minutes to run.''')
    parser.add_argument('--trials', type=int, default=100,
                        help='''Trials to run.''')
    parser.add_argument('--parallel', type=int, default=1,
                        help='''Max concurrent trials to run in parallel.''')
    parser.add_argument('--ensemble_size', type=int, default=3,
                        help='''Number of top models to include in the ensemble.''')
    args = parser.parse_args()

    if not os.path.isdir(args.output):
        os.mkdir(args.output)

    ehd_loader = EHD_Loader(args.pickle)

    summary = tune_hyperparameters(ARCHITECTURE, XTYPE, YTYPE, FILTERS,
                                   loader=ehd_loader,
                                   trials=args.trials,
                                   time_limit=args.minutes,
                                   max_concurrent=args.parallel,
                                   output_dir=args.output)

    # Enforce unique combinations of validation tasks
    ensemble = []
    eval_datasets = []
    for _ in range(args.ensemble_size):
        in_the_running = summary[summary.user_attrs_eval_datasets.apply(
            lambda x: x not in eval_datasets
        )]
        best = in_the_running.iloc[np.argmax(in_the_running.value)]
        ensemble.append(best.number)
        eval_datasets.append(best.user_attrs_eval_datasets)

    print(f'Forming ensemble from models: {ensemble}')

    ensemble_path = os.path.join(args.output, "ensemble")
    if not os.path.isdir(ensemble_path):
        os.mkdir(ensemble_path)
    for i in ensemble:
        copy2(os.path.join(args.output, f"model_{i}.pickle"), ensemble_path)


    # summaries.append(summary)

    # TODO re-do final report format
    # Print summary
    # cols = summaries[0].columns
    # best_idx = [s['value'].argmax() for s in summaries]
    # if 'user_attrs_eval_dataset' in cols:
    #     print('dataset:')
    #     print([s['user_attrs_eval_dataset'].loc[best_idx[i]] for i, s in enumerate(summaries)])
    # if 'value' in cols:
    #     print('values:')
    #     print([s['value'].loc[best_idx[i]] for i, s in enumerate(summaries)])
    # if 'user_attrs_MSE' in cols:
    #     print('MSEs:')
    #     print([s['user_attrs_MSE'].loc[best_idx[i]] for i, s in enumerate(summaries)])
    # if 'user_attrs_r' in cols:
    #     print('r:')
    #     print([s['user_attrs_r'].loc[best_idx[i]] for i, s in enumerate(summaries)])
