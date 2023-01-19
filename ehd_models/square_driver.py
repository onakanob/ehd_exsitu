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
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

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


def choose_squares_ver2(widths, v_thresh, w_thresh=1, samples=10_000, vis_output=None):
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
        EHD_Ensemble(directory="C:/Dropbox/Python/ehd_exsitu/ensembles/"\
                     + "230118 Ensemble Classifier", task='classify')
    regressor =\
        EHD_Ensemble(directory="C:/Dropbox/Python/ehd_exsitu/ensembles/"\
                     + "230118 Ensemble Regressor", task='regress')

    # Generate a large grid of potential control inputs
    length = int(np.sqrt(samples))
    Vmin = 1.9
    Vmax = 3.5
    Wmin = 2.5
    Wmax = 12

    V, W = np.meshgrid(
        np.linspace(Vmin, Vmax, num=length),
        np.logspace(np.log10(Wmin), np.log10(Wmax), num=length)
    )
    X = np.vstack([V.ravel(), W.ravel()]).T
    jets, jet_var = classifier.predict(X)
    width, w_var = regressor.predict(X)
    import ipdb; ipdb.set_trace()

    Xj = X[jets]
    jet_varj = jet_var[jets]
    widthj = width[jets]
    w_varj = w_var[jets]

    def cost(w):
        return 1 * (widthj - w)**2 \
             + 1 * jet_varj \
             + 0.6 * w_varj \
             + 0.5 * Xj[:, -1]

    waves = []
    for w in widths:
        costs = cost(w)
        waves.append(Xj[np.argmin(costs)] * np.array([v_thresh, w_thresh]))

    # TODO print space visualizer
    if vis_output is not None:
        df = pd.DataFrame({"V/Vt": X[:, 0],
                          "w/wt": X[:, 1],
                          "jets": jets,
                          "jet_var": jet_var,
                          "width": w,
                          "w_var": w_var,})
        df['width'].iloc[df['width'] < 0] = 0
        sns.set_theme(style="whitegrid")
        g = sns.relplot(
            data=df,
            x="V/Vt", y="w/wt",
            hue="jets", size="width",
            sizes=(10, 200),
        )
        plt.title("Predicted Feature Widths")
        plt.show()
        plt.clf()

        g = sns.relplot(
            data=df,
            x="V/Vt", y="w/wt",
            hue="jet_var", size="w_var",
            sizes=(10, 200),
        )
        plt.title("Estimates of prediction variability")
        plt.show()
        plt.clf()
    return waves


def choose_squares_ver1(widths, v_thresh, w_thresh=1, samples=10_000):
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
    jets, jet_var = classifier.predict(X)
    width, w_var = regressor.predict(X)

    X = X[jets]
    jet_var = jet_var[jets]
    width = width[jets]
    w_var = w_var[jets]

    def cost(w):
        return 1 * (width - w)**2 \
             + 1 * jet_var \
             + 0.6 * w_var \
             + 10 * X[:, -1]

    waves = []
    for w in widths:
        costs = cost(w)
        waves.append(X[np.argmin(costs)] * np.array([v_thresh, w_thresh]))
    return waves