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
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import pearsonr

from .utils import cell_to_array, parse_volt_strings


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
    max_offset = waves.index.max() - dots.index.max()
    for offset in range(max_offset):
        test_auac = auac.loc[dots.index + offset]
        corr, _ = pearsonr(dots.area, test_auac)
        corrs.append(corr)
    offset = np.argmax(corrs)
    print(f'Using offset {offset} for observation matching with dataset {directory}')

    dots['wave'] = np.array(waves.wave.loc[dots.index + offset])
    dots['vector'] = np.array(waves.vector.loc[dots.index + offset])
    dots['volts'] = np.array(waves.volts.loc[dots.index + offset])
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
        self.datasets = []
        for i, row in index.iterrows():
            loc = os.path.join(index_dir, row['Path'])
            wavefile = os.path.join(loc, row['Results'])  # get from json!
            # TODO read pattern_params.json for wav_file and um_per_px
            um_per_px = 1                                 # TODO do right
            # TODO delete Results columns from index files
            self.datasets.append(ehd_dir2data(loc, wavefile, um_per_px))

    def get_datasets(self):
        return self.datasets

#     # Experimental
#     waves.loc[i, 'absintegral'] = np.sum(np.abs(waves.wave.loc[i]))
#     waves.loc[i, 'absmax'] = np.max(waves.wave.loc[i])
#     waves.loc[i, 'vmag'] = np.sqrt(np.sum(waves.vector.loc[i] ** 2))
#     waves.loc[i, 'bias'] = np.abs(waves.vector.loc[i][0])
#     waves.loc[i, 'maxcumsum'] = np.max(np.abs(np.cumsum(waves.wave.loc[i])))
