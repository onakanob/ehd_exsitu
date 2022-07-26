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
from pic_parser.utils import cell_to_array


if __name__ == "__main__":
    DIR = sys.argv[1]
    # DIR = "C:/Dropbox/SPEED/Self Driving EHD/Data/2-May-2022__run 2"
    # PIC = "2-may-22 run2.bmp"
    # PATTERN = "pattern.txt"
    LOGDIR = "logs"
    # WAVE_FILE = "results 2-may-22 run2.xlsx"

    params_file = os.path.join(DIR, "pattern_params.json")
    # Positions of individual prints along the pattern path
    offsets_file = os.path.join(DIR, LOGDIR, "offsetlist.txt")
    # Points to exclude for any reason (manually edit this file)
    exclude_file = os.path.join(DIR, LOGDIR, "exclude_idx.txt")
    output_file = os.path.join(DIR, LOGDIR, "measurements.xlsx")

    with open(params_file, 'r') as f:
        pattern_params = json.load(f)
    pic_path = os.path.join(DIR, pattern_params['picture'])
    pattern_path = os.path.join(DIR, pattern_params['pattern'])
    wave_path = os.path.join(DIR, pattern_params['wave_file'])

    offsets = np.loadtxt(offsets_file)
    exclude = np.loadtxt(exclude_file).astype(int)

    patches = isolate_patches(pic_path, pattern_path, pattern_params, offsets,
                              exclude=exclude)

    waves = pd.read_excel(wave_path, index_col=0)

    # Expand waves dataframe to admit measured quantities
    new_cols = ('area', 'obj_count',
                'absintegral', 'absmax', 'vmag', 'bias', 'maxcumsum')
    for col in new_cols:
        waves[col] = None

    waves.wave = waves.wave.apply(cell_to_array)
    waves.vector = waves.vector.apply(cell_to_array)

    def log_pic(patch, desc):
        Image.fromarray(patch).save(os.path.join(
            DIR, f"logs/patches/image_{desc}.png"))

    histogram_patches(patches, bins=128, xlim=(60, 256),
                      output=os.path.join(DIR, LOGDIR,
                                          'brightness_histogram.png'))

    thresh = pattern_params["image_thresh"]
    min_size = pattern_params["image_minsize"]
    um_per_px = 1e3 / pattern_params["px_per_mm"]

    for i, patch in patches.items():
        if i in waves.index:
            area, count, contours = parse_patch(patch.copy(),
                                                threshold=thresh,
                                                min_size=min_size,
                                                return_image=True)
            log_pic(patch, i)
            log_pic(contours, f"{i}_contours_area {area}_count {count}")

            # Analytics
            waves.loc[i, 'area'] = um_per_px ** 2 * area
            waves.loc[i, 'obj_count'] = count
    
            # Experimental
            waves.loc[i, 'absintegral'] = np.sum(np.abs(waves.wave.loc[i]))
            waves.loc[i, 'absmax'] = np.max(waves.wave.loc[i])
            waves.loc[i, 'vmag'] = np.sqrt(np.sum(waves.vector.loc[i] ** 2))
            waves.loc[i, 'bias'] = np.abs(waves.vector.loc[i][0])
            waves.loc[i, 'maxcumsum'] = np.max(np.abs(np.cumsum(waves.wave.loc[i])))

    # Remove rows without a patch
    waves.drop(waves[waves.bias.isnull()].index, inplace=True)

    for key in ("absintegral", "absmax", "vmag", "bias", "maxcumsum"):
        X = waves[key]
        _, _, r_value, _, _ = linregress(
            X.astype(float), waves.area.astype(float))
        plt.plot(X, waves.area, '*')
        plt.title(f'{key} vs printed area  r: {r_value:.3f}')
        plt.savefig(os.path.join(DIR, LOGDIR, f'{key}_corr.png'))
        plt.clf()

    waves.to_excel(output_file)
