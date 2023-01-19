# -*- coding: utf-8 -*-
"""
Script to generate the waveform recipe file parameters1.txt for a 

Created 30-December-2022

@author: Oliver Nakano-Baker
"""

import os
import sys
import argparse

from ehd_models.square_driver import choose_squares_ver1, choose_squares_ver2
from gridgraph.patterns import write_ehd_params_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vt", type=int,
                        help='''V_threshold used in the print run. Nozzle
                        should be "large", "std", or "sfine")''')
    parser.add_argument("--wt", type=float, default=1.0,
                        help='''w_threshold used in the print run. Nozzle
                        should be "large", "std", or "sfine")''')
    parser.add_argument("--noz", type=str, default="std",
                        help='''nozzle type used in the print run. Nozzle
                        should be "large", "std", or "sfine")''')
    parser.add_argument("--tar", type=str,
                        help='''Tuple of the target line widths, in
                        micrometers, and waveform order, to be added to the
                        recipe file.''')
    parser.add_argument("--ver", type=int, default=-1,
                        help='''Version of model to invoke. Defualt to the most
                        recently-added model version.''')
    args = parser.parse_args()

    VERSION = [1, 2]
    if args.ver in VERSION:    # I hate this
        ver = args.ver
    else:
        ver = VERSION[args.ver]
    print(f"Using model version {ver}")

    widths = eval(args.tar)

    if ver == 1:                # Gen 1 solution
        waves = choose_squares_ver1(widths, v_thresh=args.vt, w_thresh=args.wt)
    if ver == 2:
        waves = choose_squares_ver2(widths, v_thresh=args.vt, w_thresh=args.wt,
                                    vis_output='vis_squares.png')
    else:
        raise NotImplementedError(f"generate_waves workflow {f} not implemented.")

    waves = [(widths[i], int(waves[i][0]), waves[i][1]) for i in
             range(len(widths))]
    print(f"waves: {waves}")
    write_ehd_params_file(waves)
