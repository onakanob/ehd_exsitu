# -*- coding: utf-8 -*-
"""
Use in-code variables and GUI to align patch locations with printed features in
an optical micrograph of EHD print traces.

Created on Tue Mar 22 11:20:09 2022

@author: Oliver
"""

import os
import sys
import argparse

from pic_parser.patches_gui import run_alignment_gui, run_patches_gui
from pic_parser.patch_tools import parse_patches, histogram_patches


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--dir', type=str,
                        help='''Directory containing params file, picture, and
                        pattern.''')
    parser.add_argument('--cold_start_align', action='store_true', default=False,
                        help='''[WARNING] Reset all saved parameters and run
                        pattern alignment workflow''')
    parser.add_argument('--align', action='store_true', default=False,
                        help='Run pattern alignment workflow')
    parser.add_argument('--place', action='store_true', default=False,
                        help='Run patch placement workflow')
    parser.add_argument('--parse', action='store_true', default=False,
                        help='Parse all patches workflow')
    args = parser.parse_args()

    params_file = os.path.join(args.dir, "pattern_params.json")

    if args.cold_start_align:
        run_alignment_gui(params_file=params_file, cold_start=True)
    elif args.align:
        run_alignment_gui(params_file=params_file, cold_start=False)
    elif args.place:
        run_patches_gui(params_file=params_file)
    elif args.parse:
        parse_patches(params_file=params_file)
