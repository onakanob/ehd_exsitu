# -*- coding: utf-8 -*-
"""
Toolset for generating square waves using pre-process afine transforms and
voltage- and pulse-width-threshold measurements to center the process.

Created December 2022

@author: Oliver Nakano-Baker
"""
import os
import glob

import numpy as np

from . import EHD_Model


class EHD_Ensemble():
    def __init__(self, directory, task='classify'):
        self.task = task

        # List all pickles in directory
        file_list = glob.glob(os.path.join(directory, "*.pickle"))

        # Load each into an EHD_Model
        self.models = []
        for f in file_list:
            self.models.append(EHD_Model.load(f))

    def predict(self, X):
        outputs = [m.predict(X) for m in self.models]
        outputs = np.vstack(outputs).T  # N samples by M models

        if self.task == 'classify':
            var = 1 - 2 * np.abs(0.5 - outputs.mean(1))
            jets = outputs.mean(1) > 0.5
            return jets, var
        elif self.task == 'regress':
            Y = np.zeros((len(X), outputs.shape[-1]-2))
            for i, row in enumerate(outputs):  # Drop largest and smallest
                imin = row == row.min()
                imax = row == row.max()
                Y[i] = row[~(imin + imax)]
            return Y.mean(1), Y.std(1)

        else:
            raise ValueError('task type must be "classify" or "regress"')


def choose_squares_ver1(widths, v_thresh, samples=10_000):
    """
    Load up ensembles for jetting classification and width regression.
    Generate a large grid of potential control inputs.
    For the whole grid, use each ensemble to predict whether it will jet, and
    what the feature width will be.
    For each width, select the waveform that maximizes the joint goals, from
    most to least important:
     - High confidence it will jet
     - Mean estimated width within a tolerance of the target
     - High confidence in width prediction
     - Small pulsewidth
    """
    # Load up ensembles
    classifier =\
        EHD_Ensemble(directory="C:/Dropbox/Python/ehd_exsitu/experiments/"\
                     + "230116 Ensemble Classifier", task='classify')
    regressor = EHD_Ensemble(directory="C:/Dropbox/Python/ehd_exsitu/"\
                     + "experiments/230117 Ensemble Regressor", task='regress')

    # Generate a large grid of potential control inputs
    length = int(np.sqrt(samples))
    Vmin = 1.9
    Vmax = 3.5
    Wmin = 0.01
    Wmax = 0.4
    V, W = np.meshgrid(
        np.linspace(Vmin, Vmax, num=length),
        np.logspace(np.log10(Wmin), np.log10(Wmax), num=length)
    )
    X = np.vstack([V.ravel(), W.ravel()]).T
    jets = classifier.predict(X)
    width = regressor.predict(X)

    waves = []
    for w in widths:
        waves.append([1, 1])
    return waves
