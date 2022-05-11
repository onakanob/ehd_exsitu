# -*- coding: utf-8 -*-
"""
Use in-code variables and GUI to align patch locations with printed features in
an optical micrograph of EHD print traces.

Created on Tue Mar 22 11:20:09 2022

@author: Oliver
"""

import os

from pic_parser.patches_gui import run_patches_gui


if __name__ == "__main__":
    DIR = 'C:/Dropbox/SPEED/Self Driving EHD/Data/2-May-2022__run 1'
    PIC = '2-may-22 run1.bmp'
    PATTERN = 'pattern.txt'

    picpath = os.path.join(DIR, PIC)
    patternpath = os.path.join(DIR, PATTERN)

    params_file = os.path.join(DIR, "logs/pattern_params.json")
    offsets_file = os.path.join(DIR, "logs/offsetlist.txt")

    run_patches_gui(picpath, patternpath, params_file, offsets_file)
