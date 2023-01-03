# -*- coding: utf-8 -*-
"""
Script to generate the waveform recipe file parameters1.txt for a 

Created 30-December-2022

@author: Oliver Nakano-Baker
"""

import os
import sys
import argparse

from ehd_models import choose_squares_ver1
from gridgraph.patterns import write_ehd_params_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obs", type=str,
                        help='''Tuple of (V_threshold, w_threshold, nozzle type
                        used in the print run. Nozzle should be "large", "std",
                        or "sfine")''')
    parser.add_argument("--tar", type=str,
                        help='''Tuple of the target line widths, in
                        micrometers, and waveform order, to be added to the
                        recipe file.''')
    parser.add_argument("--ver", type=int, default=-1,
                        help='''Version of model to invoke. Defualt to the most
                        recently-added model version.''')
    args = parser.parse_args()

    VERSION = [1]
    if args.ver in VERSION:    # I hate this
        ver = args.ver
    else:
        ver = VERSION[args.ver]
    print(f"Using model version {ver}")

    if ver == 1:                # Gen 1 solution
        widths = eval(args.tar)
        v_thresh, w_thresh, nozzle = eval(args.obs)
        waves = choose_squares_ver1(widths,
                                    v_thresh=v_thresh,
                                    w_thresh=w_thresh,
                                    nozzle=nozzle)
        waves = [(widths[i], waves[i][0], waves[i][1]) for i in
                 range(len(widths))]
        write_ehd_params_file(waves)
    else:
        raise NotImplementedError(f"generate_waves workflow {f} not implemented.")

    
