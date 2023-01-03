# -*- coding: utf-8 -*-
"""
Test methods for ehd_dataset.patch_tools. This follows a similar workflow to
the script parse_patches.py in the root folder.

Created on Dec 5 2022

@author: Oliver Nakano-Baker
"""

import os
import sys
import json
import pytest

import numpy as np

sys.path.append('..')
from pic_parser.patch_tools import (isolate_patches, parse_patch,
                                    histogram_patches, parse_patches)


def test_patches():
    DIR = "./patch_test_files"
    LOGDIR = "logs"

    # System-generated files
    params_file = os.path.join(DIR, "pattern_params.json")
    # Positions of individual prints along the pattern path
    offsets_file = os.path.join(DIR, LOGDIR, "offsetlist.txt")
    # Points to exclude for any reason (manually edit this file)
    exclude_file = os.path.join(DIR, "exclude_idx.txt")
    output_file = os.path.join(DIR, LOGDIR, "measurements.xlsx")

    # Inputs/user-generated files
    with open(params_file, 'r') as f:
        pattern_params = json.load(f)
    pic_path = os.path.join(DIR, pattern_params['picture'])
    pattern_path = os.path.join(DIR, pattern_params['pattern'])


    offsets = np.loadtxt(offsets_file)
    exclude = np.loadtxt(exclude_file).astype(int)

    patches = isolate_patches(pic_path, pattern_path, pattern_params, offsets,
                              exclude=exclude)
    assert len(patches.keys()) == 5

    output_file = os.path.join(DIR, LOGDIR, 'brightness_histogram.png')
    histogram_patches(patches, bins=128, xlim=(60, 256),
                      output=output_file)
    assert os.path.isfile(output_file)

    um_per_px = 1e3 / pattern_params["px_per_mm"]

    for i, patch in patches.items():
        properties, contours = parse_patch(patch.copy(),
                                    threshold=pattern_params["image_thresh"],
                                    low_thresh=pattern_params.get("image_lowthresh"),
                                    min_size=pattern_params["image_minsize"],
                                    um_per_px=um_per_px,
                                    return_image=True)
        break
    assert "mean_width" in properties.keys()
    assert contours.dtype == "uint8"


def test_parser():
    DIR = "./patch_test_files"
    params_file = os.path.join(DIR, "pattern_params.json")
    parse_patches(params_file=params_file, test=True)


if __name__ == "__main__":
    test_parser()
    test_patches()
