# -*- coding: utf-8 -*-
"""
Dataloader for compiling and preprocessing input waveforms and output print
feature metrics from multiple electrohydrodynamic inkjet experimental
runs. Deliver all data in a dataframe for machine learning activities.

Created on June 11 2022

@author: Oliver Nakano-Baker
"""

import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# from scipy.stats import pearsonr

from .utils import cell_to_array, parse_volt_strings, correlate_dfs


def ehd_dir2data(directory, wavefile, um_per_px):
    MEAS_PATH = 'logs/measurements.xlsx'
    VOLT_GAIN = 300

    waves = pd.read_excel(wavefile, index_col=0)
    waves['volts'] = waves.note.apply(parse_volt_strings)
    waves['volts'] *= VOLT_GAIN
    waves['wave'] = waves.wave.apply(cell_to_array)
    waves['wave'] *= waves.volts
    waves['vector'] = waves.vector.apply(cell_to_array)
    waves['vector'] *= waves.volts

    dots = pd.read_excel(os.path.join(directory, MEAS_PATH), index_col=0)

    # area under absolute waveform
    auac = waves.wave.apply(lambda x: np.sum(np.abs(x)))

    corrs = []
    max_offset = 20
    for offset in range(max_offset):
        corr, _, _ = correlate_dfs(dots, 'area', auac, None, offset)
        corrs.append(corr)
    offset = np.argmax(corrs)
    _, Aidx, Bidx = correlate_dfs(dots, 'area', auac, None, offset)
    print(f'dataset {directory}\toffset {offset}\tcorr {np.max(corrs)}')

    dots = dots.loc[Aidx]

    dots['wave'] = np.array(waves.wave.loc[Bidx])
    dots['vector'] = np.array(waves.vector.loc[Bidx])
    dots['volts'] = np.array(waves.volts.loc[Bidx])
    dots['area'] *= um_per_px ** 2

    # Dev validation code:
    # plt.plot(corrs)
    # plt.title(f'best correlation offset: {offset}')
    # plt.show()

    # corr, _ = pearsonr(dots.area, dots.wave.apply(lambda x: np.sum(np.abs(x))))
    # print(corr)
    # corr, _ = pearsonr(dots.area, dots.vector.apply(lambda x: np.sqrt(np.sum(x**2))))
    # print(corr)

    return dots


class EHD_Loader():
    """Class for loading multiple waveform-print metric pairs by referencing an
    index file. For each dataset, match up image analysis and waveform indices,
    pre-process fields to be numerical numpy arrays, and provide dataframes on
    demand."""
    def __init__(self, index_file):
        index_dir = str(Path(index_file).parent)
        index = pd.read_excel(index_file)
        self.names = []
        self.datasets = []

        for i, row in index.iterrows():
            try:
                loc = os.path.join(index_dir, row['Path'])
                with open(os.path.join(loc, 'pattern_params.json'), 'r') as f:
                    params = json.load(f)
                wavefile = os.path.join(loc, params['wave_file'])
                um_per_px = 1e3 / params['px_per_mm']
                self.datasets.append(ehd_dir2data(loc, wavefile, um_per_px))
                self.names.append(os.path.basename(row['Path']))
            except Exception as e:
                print(f"Failed to load {row['Path']}: {e}")
                # raise(e)        # TODO

    def get_datasets(self):
        return self.datasets

    def x_method_vector(self, df):
        """Method for grabbing the vector description from a dataframe and
        returning it as a numpy array."""
        return np.array(list(df.vector))

    def y_method_area(self, df):
        """Method for returning the feature area from a dataframe as a numpy
        array"""
        import ipdb; ipdb.set_trace()
        return None

    def folded_dataset(self, fold, xtype, ytype):
        if not type(fold) == int:
            raise TypeError("fold must be an integer")
        elif fold >= len(self.names):
            raise ValueError(f"not enough EHD datasets to create fold {fold}")
        train_set = {'X': np.array([[]]), 'Y': np.array([[]]), 'p': np.array([])}
        eval_set =  {'X': np.array([[]]), 'Y': np.array([[]]), 'p': np.array([])}
        fold_name = self.names[fold]

        if xtype == "vector":
            xmethod = self.x_method_vector
        else:
            raise ValueError(f"EHD dataset with xtype {xtype} not implemented")

        if ytype == "area":
            ymethod = self.y_method_area
        else:
            raise ValueError(f"EHD dataset with ytype {ytype} not implemented")

        for p in range(len(self.names)):
            if fold == p:
                eval_set['X'] = np.concatenate((eval_set['X'],
                                                xmethod(self.datasets[p])))
                eval_set['Y'] = np.concatenate((eval_set['Y'],
                                                ymethod(self.datasets[p])))
                eval_set['p'] = np.concatenate((eval_set['p'],
                                                p * np.ones(len(datasets[p]))))
            else:
                train_set['X'] = np.concatenate((train_set['X'],
                                                 xmethod(self.datasets[p])))
                train_set['Y'] = np.concatenate((train_set['Y'],
                                                 ymethod(self.datasets[p])))
                train_set['p'] = np.concatenate((train_set['p'],
                                                p * np.ones(len(datasets[p]))))

        return train_set, eval_set, fold_name


    def __repr__(self):
        return f"EHD dataset loader with run directories: {self.names}"

    @property
    def num_folds(self):
        return len(self.names)


#     # Experimental
#     waves.loc[i, 'absintegral'] = np.sum(np.abs(waves.wave.loc[i]))
#     waves.loc[i, 'absmax'] = np.max(waves.wave.loc[i])
#     waves.loc[i, 'vmag'] = np.sqrt(np.sum(waves.vector.loc[i] ** 2))
#     waves.loc[i, 'bias'] = np.abs(waves.vector.loc[i][0])
#     waves.loc[i, 'maxcumsum'] = np.max(np.abs(np.cumsum(waves.wave.loc[i])))
