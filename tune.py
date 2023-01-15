# -*- coding: utf-8 -*-
"""
Created 12-January-2023

Tune hyperparameters for any model with a .tune() method.

@author: Oliver Nakano-Baker
"""
import argparse

import numpy as np
from ray import tune

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

# Define the hyperparameter search space for this model
PARAMS = {'n_estimators': tune.lograndint(10, 1000),
          'max_depth': tune.lograndint(1, 100),
          'min_samples_split': tune.randint(2, 10),
          'min_samples_leaf': tune.randint(1, 4),
          'max_features': tune.randint(1, 3),
          'max_leaf_nodes': tune.lograndint(2, 100),
          'bootstrap': tune.choice((False, True)),
          'max_samples': tune.randint(10, 50)}


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default="./TUNE_OUTPUT",
                        help='''Directory to output results and save
                        models.''')
    parser.add_argument('--index', type=str,
        default="C:/Dropbox/SPEED/Self Driving EHD/Datasets/dataset_index.xlsx",
                        help='''Path to dataset index excel file.''')
    args = parser.parse_args()

    ehd_loader = EHD_Loader(args.index)
    tune_hyperparameters(ARCHITECTURE, XTYPE, YTYPE, FILTERS, PARAMS,
                         loader=ehd_loader, output_dir=args.output)
