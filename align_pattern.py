# -*- coding: utf-8 -*-
"""
Use in-code variables and GUI to align patch locations with printed features in
an optical micrograph of EHD print traces.

Created on Tue Mar 22 11:20:09 2022

@author: Oliver
"""

import os
import sys
import json

from pic_parser.patches_gui import run_alignment_gui


if __name__ == "__main__":
    DIR = sys.argv[1]

    params_file = os.path.join(DIR, "pattern_params.json")

    with open(params_file, 'r') as f:
        pattern_params = json.load(f)
    pic = pattern_params['picture']

    picpath = os.path.join(DIR, pattern_params['picture'])
    patternpath = os.path.join(DIR, pattern_params['pattern'])

    run_alignment_gui(picpath, patternpath, params_file)
