# -*- coding: utf-8 -*-
"""
Test GUI method for placing patches.

Created on Dec 5 2022

@author: Oliver Nakano-Baker
"""

import os
import sys
import json
import pytest

sys.path.append('..')
from pic_parser.patches_gui import run_patches_gui, run_alignment_gui


def test_patches_gui():
    app = run_patches_gui(im_path="./patch_test_files/2-may-22 run1.bmp",
                          pattern_path="./patch_test_files/pattern.txt",
                          params_file="./patch_test_files/pattern_params.json",
                          test=True)
    assert app.pic.dtype == "uint8"
    assert type(app.params['theta']) is int
    assert len(app.pattern) == 36

    params = app.get_patch_params(10)
    assert params[2] == 225     # image_thresh
    app.set_patch_params({'include_me': params[0],
                          'offset': params[1]}, loc=10)
    app.destroy()


def test_align_gui():
    app = run_alignment_gui(im_path="./patch_test_files/2-may-22 run1.bmp",
                            pattern_path="./patch_test_files/pattern.txt",
                            params_file="./patch_test_files/pattern_params.json",
                            test=True)
    assert app.pic.dtype == "uint8"
    assert type(app.params['theta']) is int
    assert len(app.pattern) == 36
    app.destroy()


if __name__ == "__main__":
    test_patches_gui()
    test_align_gui()

