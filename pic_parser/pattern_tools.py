# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 10:54:36 2022

@author: Oliver
"""
import math as m
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt


def record_to_vec(df, i):
    wave = df.iloc[i].wave[1:-1].split()
    vec = df.iloc[i].vector[1:-1].split()
    return np.array(vec).astype(float), np.array(wave).astype(float)


def cumulative_distance(pattern):
    """Calculate the cumulative distance traveled at each point in the pattern.
    """
    dists = np.zeros(pattern.shape[0])
    for i, x in enumerate(pattern):
        if i:
            segment_length = np.linalg.norm(x - pattern[i - 1])
            dists[i] = dists[i - 1] + segment_length
        else:
            pass
    return dists


def microns_into_pattern(x, pattern, scale):
    """Return the point coordinates of the location 'x' microns into the
    provided pattern. Pattern assumed in units pixels, x in units microns.
    """
    dists = cumulative_distance(pattern) / scale
    idx_past = np.where(dists > x)[0].min()  # first point further than x
    x_rel = (x - dists[idx_past - 1]) / (dists[idx_past] - dists[idx_past - 1])
    vec = pattern[idx_past] - pattern[idx_past - 1]
    point = pattern[idx_past - 1] + x_rel * vec
    angle = vec / np.linalg.norm(vec)
    return point, np.arctan2(angle[1], angle[0])


def old_patchmaker(img, height, width, center_y, center_x, angle):
    """Courtesy user Rozen Moon at
    https://stackoverflow.com/questions/49892205/extracting-patch-of-a-certain-size-at-certain-angle-with-given-center-from-an-im
    """
    theta = angle/180*3.14
    img_shape = np.shape(img)
    # print(img_shape)
    x, y = np.meshgrid(range(img_shape[1]), range(img_shape[0]))
    rotatex = x[center_y-m.floor(height/2):center_y+m.floor(height/2),
                center_x-m.floor(width/2):center_x+m.floor(width/2)]
    rotatey = y[center_y-m.floor(height/2):center_y+m.floor(height/2),
                center_x-m.floor(width/2):center_x+m.floor(width/2)]
    coords = [rotatex.reshape((1, height*width))-center_x,
              rotatey.reshape((1, height*width))-center_y]
    coords = np.asarray(coords)
    coords = coords.reshape(2, height*width)
    roatemat = [[m.cos(theta), m.sin(theta)], [-m.sin(theta), m.cos(theta)]]
    rotatedcoords = np.matmul(roatemat, coords)
    patch = ndimage.map_coordinates(
        img, [rotatedcoords[1]+center_y, rotatedcoords[0]+center_x], order=1, mode='nearest').reshape(height, width)
    return patch


def patchmaker(img, height, width, center_y, center_x, angle):
    new_height = np.abs(np.cos(angle) * height + np.sin(angle) * width)
    new_width = np.abs(np.cos(angle) * width + np.sin(angle) * height)
    max_vals = np.shape(img)
    min_x = np.max((0, int(center_x - new_width / 2)))
    max_x = np.min((max_vals[1], int(center_x + new_width / 2)))
    min_y = np.max((0, int(center_y - new_height / 2)))
    max_y = np.min((max_vals[0], int(center_y + new_height / 2)))
    patch = img[min_y:max_y, min_x:max_x]
    return patch



def align_pattern(csv, scale, theta, offset):
    R = np.array([[np.cos(theta), -np.sin(theta)],
                  [np.sin(theta),  np.cos(theta)]])

    # Import pattern and remove non-printed orders
    pattern = np.genfromtxt(csv, delimiter="\t")
    pattern = pattern[:, :2][pattern[:, -1] == 1]

    # Affine transformations
    pattern = np.matmul(pattern, scale * R)
    pattern += np.array(offset)

    return pattern


def list_points_on_pattern(pattern, start, spacing, point_count, scale):
    points = [microns_into_pattern(
        start + i * spacing, pattern, scale) for i in range(point_count)]
    return points


def display_pattern(im, pattern, point=None, axs=None):
    if axs is None:
        _, axs = plt.subplots(1, 1)

    axs.imshow(im)
    axs.plot(pattern[:, 1], pattern[:, 0], '--r', linewidth=0.2)

    # point, angle = points[point_idx]
    if point is not None:
        point = np.array(point)
        if len(point.shape) == 1:
            point = point[None, :]
        for p in point:
            axs.plot(p[1], p[0], '*r')
    return axs


def display_patch(im, point, angle, spacing, pitch, pattern, axs=None):
    if axs is None:
        _, axs = plt.subplots(1, 2)

    display_pattern(im, pattern, point=point, axs=axs[0])

    patch = patchmaker(im, spacing, pitch,
                       int(point[0]), int(point[1]), angle)
    axs[1].imshow(patch)
    # plt.show()
    return axs
