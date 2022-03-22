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
    DIR = "E:\Dropbox\SPEED\Self Driving EHD\Data\Olympus mosaics"

    PIC = os.path.join(DIR, "10-mar-22__1.6V harmonics__10x.tif")
    PATTERN = os.path.join(DIR, "pattern.txt")
    LOG = os.path.join(DIR, "results 10 mar large noz 300 pitch.xlsx")

    offsets_file = os.path.join(DIR, "logs/offsetlist.txt")

    run_patches_gui(PIC, PATTERN, offsets_file)
