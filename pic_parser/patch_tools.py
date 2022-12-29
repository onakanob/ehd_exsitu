# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 11:53:09 2022

@author: Oliver
"""
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import cv2 as cv
from PIL import Image
from PIL.ImageOps import grayscale

from .pattern_tools import patchmaker, align_pattern, microns_into_pattern
from .param_tools import Params_Reader


def histogram_patches(patches, bins=100, xlim=(140, 200), output=None):
    """
    Display histogram of brightness values from all entries in a collection of images

    patches : DICT
        Values are numpy int arrays.
    """
    brightness = np.array([])
    for patch in patches.values():
        brightness = np.concatenate((brightness,
                                     patch.ravel()))
    plt.hist(brightness, bins=bins)
    plt.xlim(xlim)
    if output is None:
        plt.show()
    else:
        plt.savefig(output)
    plt.clf()


def dim3_dist(A, B):
    diff_squared = (A - B)**2
    min_dist = np.sqrt(diff_squared.sum(2)).min(-1)
    return min_dist


def silver_color_filt(I):
    """These are color samples taken from the 8-Aug-2022 run, with colors from
    the background and from ink spots. Return a per-pixel measure of distance
    from background or ink color centers. 0=background, 1=ink."""
    I = np.array(I)
    background_rgb = np.array([[170, 178, 204],
                               [170, 175, 203],
                               [183, 196, 226],
                               [180, 185, 213],
                               [192, 203, 230],
                               [191, 197, 219],
                               [0, 0, 0]]).T
    b_dist = dim3_dist(I[:, :, :, None], background_rgb[None, None, :, :])
    ink_rgb = np.array([[190, 212, 215],
                        [199, 216, 216],
                        [188, 153, 88],
                        [235, 209, 150],
                        [201, 194, 157],
                        [186, 165, 117],
                        [127, 112, 85],
                        [162, 153, 107],
                        [116, 60, 43],
                        [163, 200, 190],
                        [184, 201, 183],
                        [129, 102, 61]]).T
    i_dist = dim3_dist(I[:, :, :, None], ink_rgb[None, None, :, :])
    return (255 * b_dist/(b_dist+i_dist)).astype(int)


def open_and_preprocess_pic(picture, imfilt="grayscale"):
    if imfilt == "grayscale":
        filt_function = grayscale
    elif imfilt == "silver":
        filt_function = silver_color_filt
    else:
        raise ValueError(f"Improper imfilt function specified: {imfilt}")

    pic = Image.open(picture)
    pic = np.fliplr(np.flipud(np.array(filt_function(pic)).T))
    return pic


def isolate_patches(picture, pattern_file, pattern_params, offsets,
                    exclude=[]):
    # pic = Image.open(picture)
    # pic = np.fliplr(np.flipud(np.array(grayscale(pic)).T))
    pic = open_and_preprocess_pic(picture)
    pattern = align_pattern(pattern_file,
                            pattern_params['px_per_mm'] / 1e4,
                            pattern_params['theta'],
                            pattern_params['pattern_offset'])

    patches = {}
    for i, offset in enumerate(offsets):
        if not i in exclude:
            point, angle = microns_into_pattern(
                offset, pattern, pattern_params['px_per_mm'] * 1e-3)
            patch = patchmaker(pic,
                               height=pattern_params['spacing']\
                                   * pattern_params['px_per_mm'] * 1e-3 * 1.2,
                               width=pattern_params['pitch']\
                                   * pattern_params['px_per_mm'] * 1e-3 * 1.5,
                               center_y=int(point[0]),
                               center_x=int(point[1]),
                               angle=angle)
            patches[i] = patch
    return patches


def parse_patch(patch, threshold=170, low_thresh=None, min_size=6, um_per_px=1,
                return_image=False, DEV=False):  # TODO
    """
    Given a patch containing an image of one EHD printing experiment, parse the
    image and return measurements of the printed pattern.
    INPUTS:
    patch - grayscale image to be parsed
    threshold - intensity value above which pixels are assumed to be ink
    low_thresh - intensity value _below_ which pixels are assumed
        to be ink. (This creates a band of intensity values associated with
        background.)
    min_size - smallest number of pixels within an isolated patch that will be
        considered part of the printed pattern, for denoising
    um_per_px - microns per pixel length. Default value one will just return
        measurements in units [pixels].
    return_image - modifies the return interface to include an altered version
        of the patch that outlines the identified ink regions of the image.

    MEASUREMENTS:
    area - total area of ink in the image [um**2]
    obj_count - number of separated regions of ink
    """
    bw = (patch >= threshold).astype("uint8")
    if low_thresh is not None:
        bw = bw + (patch <= low_thresh).astype("uint8")
        

    contours, hierarchy = cv.findContours(
        bw, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    if not hierarchy is None:
        # Remove inclusions, small features, and features on edges
        top_hier = hierarchy.squeeze(
            0)[:, -1] == -1  # must not have parent
        good_size = np.array([cntr.shape[0]
                             for cntr in contours]) >= min_size
        boxes = [cv.boundingRect(cntr) for cntr in contours]  # x, y, w, h
        no_touch = ~np.array([box_touches_edge(box, patch.shape)
                             for box in boxes])

        # This dictates that each contour must:
        #     1. Not be fully within another contour
        #     2. Be at or above the minimum size limit
        #     3. Not abut the edge of the patch
        allowed = top_hier * good_size * no_touch
        contours = [cnt for j, cnt in enumerate(contours) if allowed[j]]

    properties = {}

    if len(contours):
        # cv.minAreaRect returns: (loc), (size), angle
        rect = cv.minAreaRect(np.vstack(contours))
    else:
        rect = ([0,0], [0,0], 0)
    bounding_size = [rect[1]]

    # calculate properties
    properties['obj_count'] = len(contours)
    properties['area'] = um_per_px ** 2 \
        * np.sum([cv.contourArea(cnt) for cnt in contours])  # microns^2
    properties['print_length'] = um_per_px * np.max(bounding_size)
    properties['max_width'] = um_per_px * np.min(bounding_size)
    properties['mean_width'] = safe_divide(properties['area'],
                                           properties['print_length'])

    if return_image:
        box = np.int0(cv.boxPoints(rect))
        contour_pic = cv.drawContours(patch, contours, -1, 0, 2)
        contour_pic = cv.drawContours(contour_pic, [box], -1, 255, 1)
        return properties, contour_pic
        # return properties, cv.drawContours(patch, contours, -1, 0, 2)
    elif DEV:                   # TODO
        return properties, (bw, contours)
    return properties


def box_touches_edge(box, imshape):
    x, y, w, h = box
    hh, ww = imshape
    return (x <= 0) or (y <= 0) or (x+w >= ww) or (y+h >= hh)


def safe_divide(a, b):
    if (a == 0) and (b == 0):
        return 0
    return a / b


def parse_patches(params_file, test=False):
    OUTFILE_NAME = 'measurements.xlsx'
    directory = os.path.dirname(params_file)
    output_file = os.path.join(directory, OUTFILE_NAME)

    reader = Params_Reader(params_file)

    offsets = [reader.retrieve(i, 'offset') for i in
               range(reader.params['point_count'])]

    patches = isolate_patches(picture=reader.pic_path(),
                              pattern_file=reader.pattern_path(),
                              pattern_params=reader.params,
                              offsets=offsets)

    measurements = pd.DataFrame()

    def log_pic(patch, desc):
        Image.fromarray(patch).save(os.path.join(
            directory, f"patches/image_{desc}.png"))

    um_per_px = 1e3 / reader.params["px_per_mm"]

    print(f"Parsing patches from image {reader.pic_path()}")

    for i, patch in patches.items():
        if reader.retrieve(i, 'include_me'):
            properties, contours = parse_patch(patch.copy(),
                        threshold=reader.retrieve(i, 'image_thresh'),
                        low_thresh=reader.retrieve(i, 'image_lowthresh'),
                        min_size=reader.retrieve(i, 'image_minsize'),
                        um_per_px=um_per_px,
                        return_image=True)

            log_pic(contours, i)  # TODO why sometimes mult by 255?
            # f"{i}_contours_area {properties['area']:.1f}_count {properties['obj_count']}")
    
            for key, value in properties.items():
                measurements.loc[i, key] = value
        if test and (i > 2):
            break

    measurements.to_excel(output_file)

    print(f"Logged measurements to {output_file}")
