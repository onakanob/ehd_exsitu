# -*- coding: utf-8 -*-
"""
Toolset for generating square waves using pre-process afine transforms and
voltage- and pulse-width-threshold measurements to center the process.

Created December 2022

@author: Oliver Nakano-Baker
"""


def choose_squares_ver1(widths, v_thresh, w_thresh, nozzle):
    waves = []
    for w in widths:
        waves.append((w * v_thresh / 2, w * w_thresh))
    return waves
