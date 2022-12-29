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


def ehd_dir2data(directory, wavefile, um_per_px, max_offset=140):
    MEAS_PATH = 'measurements.xlsx'
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
    for offset in range(max_offset):
        corr, _, _ = correlate_dfs(dots, 'area', auac, None, offset)
        corrs.append(corr)
    offset = np.argmax(corrs)
    _, Aidx, Bidx = correlate_dfs(dots, 'area', auac, None, offset)

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
    print(f'dataset {directory}\t{len(dots)} points\toffset {offset}\tcorr {np.max(corrs)}')

    return dots


def safecat(arr1, arr2):
    """Concatenate two arrays with numpy concatenate. If one array is None,
    simply return the other array."""
    if (arr1 is None) or (len(arr1) == 0):
        return arr2
    elif (arr2 is None) or (len(arr2) == 0):
        return arr1
    else:
        try:
            return np.concatenate((arr1, arr2))
        except:
            import ipdb; ipdb.set_trace()


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
                df = ehd_dir2data(loc, wavefile, um_per_px)
                df['jetted'] = df['area'] > 0
                self.datasets.append(df)
                self.names.append(os.path.basename(row['Path']))
            except Exception as e:
                print(f"Failed to load {row['Path']}: {e}")

    def get_datasets(self):
        return self.datasets

    def dataset_col_to_vec(self, col, p, valid_pairs_only=False):
        """Return column 'col' of dataset index 'p' as a flat numpy array"""
        if valid_pairs_only:
            index = self.pair_indices(p)
            return np.array(list(self.datasets[p][col][index]))
        return np.array(list(self.datasets[p][col]))

    def pair_indices(self, p):
        """Return indices in dataset p where a valid point exists at index
        i-1"""
        index = self.datasets[p].index
        valid = [i-1 in index for i in index]
        return index[valid]

    def last_pair_method(self, col, ytype, p):
        index = self.pair_indices(p)
        last_x = np.array(list(self.datasets[p][col][index-1]))
        last_y = np.array(list(self.datasets[p][ytype][index-1]))
        if len(last_y.shape) == 1:
            last_y = last_y[:, None]
        this_x = np.array(list(self.datasets[p][col][index]))
        return np.concatenate((last_x, last_y, this_x), axis=1)

    def filters_2_mask(self, filters, p, valid_pairs_only=False):
        if valid_pairs_only:
            index = self.pair_indices(p)
            df = self.datasets[p].loc[index]
        else:
            df = self.datasets[p]

        mask = np.ones(len(df)).astype(bool)
        for filt in filters:
            mask *= (df[filt[0]].apply(filt[1]) == filt[2]).astype(bool)
        return np.array(mask)

    def folded_dataset(self, fold, xtype, ytype, pretrain=True, filters=[]):
        """
        filters - list of 3ples, each (column, function, value). Only returns
        rows of datasets where df[column].apply(function) == value for all
        3ples in the filter list."""

        idx = self.folded_idx(filters)
        # eval dataset is element 'fold' of the filtered idx list
        fold = int(idx[fold])

        if fold >= len(self.names):
            raise ValueError(f"not enough EHD datasets to create fold {fold}")
        fold_name = self.names[fold]

        valid_pairs_only = False

        if xtype in ("vector", "wave"):  # just grab a column
            xmethod = lambda p: self.dataset_col_to_vec(col=xtype, p=p)
        elif xtype == "last_wave":  # last wave + y pair, plus x
            xmethod = lambda p: self.last_pair_method(col='wave', ytype=ytype, p=p)
            valid_pairs_only = True
        elif xtype == "last_vector":  # last wave + y pair, plus x
            xmethod = lambda p: self.last_pair_method(col='vector', ytype=ytype, p=p)
            valid_pairs_only = True
        else:
            raise ValueError(f"EHD dataset with xtype {xtype} not implemented")

        if ytype in ("area", "obj_count", "jetted"):  # just grab a column
            ymethod = lambda p: self.dataset_col_to_vec(col=ytype, p=p,
                                                        valid_pairs_only=valid_pairs_only)
        # elif ytype == "jetted":  # Did anything print at all?
        #     def jetted(p):
        #         if valid_pairs_only:
        #             index = self.pair_indices(p)
        #             y = np.array(list(self.datasets[p]['area'][index] > 0))
        #         else:
        #             y = np.array(list(self.datasets[p]['area'] > 0))
        #         # return y[:, None]
        #         return y
        #     ymethod = jetted
        elif ytype == "jetted_selectors":
            def jetted_select(p):
                y = self.dataset_col_to_vec(col='jetted', p=p,
                                            valid_pairs_only=valid_pairs_only)
                # if valid_pairs_only:
                #     index = self.pair_indices(p)
                #     y = np.array(list(self.datasets[p]['area'][index] > 0))
                # else:
                #     y = np.array(list(self.datasets[p]['area'] > 0))
                return np.concatenate((y[:, None], ~y[:, None]), axis=1)
            ymethod = jetted_select
        else:
            raise ValueError(f"EHD dataset with ytype {ytype} not implemented")


        train_set = {'X': None, 'Y': None, 'p': None}
        eval_set = train_set.copy()
        for p in idx:
            mask = self.filters_2_mask(filters, p, valid_pairs_only=valid_pairs_only)
            # if mask.sum() == 0:
            #     import ipdb; ipdb.set_trace()

            if fold == p:
                eval_set['X'] = safecat(eval_set['X'],
                                        xmethod(p)[mask])
                eval_set['Y'] = safecat(eval_set['Y'],
                                        ymethod(p)[mask])
                eval_set['p'] = safecat(eval_set['p'],
                                        p * np.ones(mask.sum()))

            elif pretrain:
                train_set['X'] = safecat(train_set['X'],
                                         xmethod(p)[mask])
                train_set['Y'] = safecat(train_set['Y'],
                                         ymethod(p)[mask])
                train_set['p'] = safecat(train_set['p'],
                                         p * np.ones(mask.sum()))

        return train_set, eval_set, fold_name


    def __repr__(self):
        return f"EHD dataset loader with run directories: {self.names}"

    def folded_idx(self, filters=None):
        if filters is None:
            return np.arange(len(self.names))

        data_select = np.ones(len(self.names))
        for filt in filters:
            for p in range(len(self.names)):
                data_select[p] *= self.filters_2_mask(filters, p).max()
        return np.where(data_select)[0]

    def num_folds(self, filters=None):
        if filters is None:
            return len(self.names)
        return len(self.folded_idx(filters))

