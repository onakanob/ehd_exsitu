# -*- coding: utf-8 -*-
"""
Parse each patch from a file using existing pattern and patch alignment
artifacts.

Created on Tue Mar 22 2022

@author: Oliver
"""

import os
import sys
import json

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import linregress
from PIL import Image

from pic_parser.patch_tools import (isolate_patches, parse_patch,
                                    histogram_patches)


if __name__ == "__main__":
    DIR = sys.argv[1]
    LOGDIR = "logs"

    params_file = os.path.join(DIR, "pattern_params.json")
    # Positions of individual prints along the pattern path
    offsets_file = os.path.join(DIR, LOGDIR, "offsetlist.txt")
    # Points to exclude for any reason (manually edit this file)
    exclude_file = os.path.join(DIR, "exclude_idx.txt")
    output_file = os.path.join(DIR, LOGDIR, "measurements.xlsx")

    with open(params_file, 'r') as f:
        pattern_params = json.load(f)
    pic_path = os.path.join(DIR, pattern_params['picture'])
    pattern_path = os.path.join(DIR, pattern_params['pattern'])

    offsets = np.loadtxt(offsets_file)
    exclude = np.loadtxt(exclude_file).astype(int)

    patches = isolate_patches(pic_path, pattern_path, pattern_params, offsets,
                              exclude=exclude)

    cols = ('area', 'obj_count')
    measurements = pd.DataFrame(columns=cols)

    def log_pic(patch, desc):
        Image.fromarray(patch).save(os.path.join(
            DIR, f"logs/patches/image_{desc}.png"))

    histogram_patches(patches, bins=128, xlim=(60, 256),
                      output=os.path.join(DIR, LOGDIR,
                                          'brightness_histogram.png'))

    um_per_px = 1e3 / pattern_params["px_per_mm"]

    for i, patch in patches.items():
        # image_lowthresh is optional - .get() returns None if doesn't exist
        area, count, contours = parse_patch(patch.copy(),
                                    threshold=pattern_params["image_thresh"],
                                    low_thresh=pattern_params.get("image_lowthresh"),
                                    min_size=pattern_params["image_minsize"],
                                    return_image=True)
        log_pic(patch, i)
        log_pic(contours, f"{i}_contours_area {area}_count {count}")

        # Analytics
        measurements.loc[i, 'area'] = um_per_px ** 2 * area
        measurements.loc[i, 'obj_count'] = count

    measurements.to_excel(output_file)
