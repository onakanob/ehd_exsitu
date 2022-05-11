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


def parse_patch(patch, threshold=170, min_size=6, return_image=False):
    bw = (patch >= threshold).astype("uint8")

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
        allowed = top_hier * good_size * no_touch  # AND
        contours = [cnt for j, cnt in enumerate(contours) if allowed[j]]

    # calculate properties
    count = len(contours)
    area = np.sum([cv.contourArea(cnt) for cnt in contours])  # pixels

    if return_image:
        return area, count, cv.drawContours(patch, contours, -1, 0, 2)
    return area, count


def box_touches_edge(box, imshape):
    x, y, w, h = box
    hh, ww = imshape
    return x <= 0 or y <= 0 or x+w >= ww or y+h >= hh
