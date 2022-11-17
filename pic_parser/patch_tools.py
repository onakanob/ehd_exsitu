# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 11:53:09 2022

@author: Oliver
"""
import numpy as np
from matplotlib import pyplot as plt
import cv2 as cv
from PIL import Image
from PIL.ImageOps import grayscale

from .pattern_tools import patchmaker, align_pattern, microns_into_pattern


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


def isolate_patches(picture, pattern_file, pattern_params, offsets,
                    exclude=[]):
    pic = Image.open(picture)
    pic = np.fliplr(np.flipud(np.array(grayscale(pic)).T))

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
                               height=pattern_params['spacing'],
                               width=pattern_params['pitch'],
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
    return x <= 0 or y <= 0 or x+w >= ww or y+h >= hh


def safe_divide(a, b):
    if (a == 0) and (b == 0):
        return 0
    return a / b
