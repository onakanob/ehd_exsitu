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
ARCHITECTURE = "v_normed_RF_class"
XTYPE = "v_normed_squares"  # vector, normed_squares, v_normed_squares
YTYPE = "jetted"            # jetted, max_width
FILTERS = [
           ('vector', lambda x: len(x), 2),
           ('Wavegen', lambda x: x, 'square'),
           ('V Thresh [V] @ .5s', np.isnan, False),
           ('SIJ Tip', lambda x: x, 'Std'),
           ('clogging', lambda x: x, False)
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
    parser.add_argument('--fold', type=int, default=0,
                        help='''Which dataset fold to use.''')
    args = parser.parse_args()

    ehd_loader = EHD_Loader(args.index)

    tune_hyperparameters(ARCHITECTURE, XTYPE, YTYPE, FILTERS,
                         loader=ehd_loader,
                         fold=args.fold,
                         trials=args.trials,
                         time_limit=args.minutes,
                         max_concurrent=args.parallel,
                         output_dir=args.output)
