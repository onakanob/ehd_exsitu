# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 00:46:17 2022

@author: Oliver
"""
import os

import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image
from PIL.ImageOps import grayscale

from .pattern_tools import (microns_into_pattern, align_pattern, display_patch)


class Micros_GUI(tk.Tk):
    px_per_mm = 795  # 10x Olympus: 795

    scale = px_per_mm / 1e4  # SIJ patterns use units 1e-4 mm
    pattern_offset = [270, 310]
    theta = np.radians(1.4)  # degrees clockwise

    point_count = 112

    # microns:
    pitch = 300
    spacing = 500
    starting_offset = 16.52e3

    def __init__(self):
        super().__init__()


class Align_GUI(Micros_GUI):  # TODO
    def __init__(self):
        super().__init__()

        self.title("Pattern Aligner")
        self.geometry("1000x1000")
        self.createWidgets()

    def createWidgets(self):
        f0 = tk.Frame()
        self.fig = plt.figure(figsize=(16, 16))
        self.fig.add_subplot(111)
        canvas = FigureCanvasTkAgg(self.fig, f0)
        canvas._tkcanvas.pack(fill=tk.BOTH, expand=1)
        f0.pack(fill=tk.BOTH, expand=1)


class Patches_GUI(Micros_GUI):
    def __init__(self, picture, pattern_file,
                 offsets_file='offsetlist.txt'):
        super().__init__()

        self.offsets_file = offsets_file

        self.pic = Image.open(picture)
        self.pic = np.fliplr(np.flipud(np.array(grayscale(self.pic)).T))
        self.pattern = align_pattern(pattern_file, self.scale, self.theta,
                                     self.pattern_offset)

        self.title("Patches Selector")
        self.geometry("1200x900")

        if os.path.isfile(offsets_file):
            self.offsets = np.loadtxt(offsets_file)
        else:
            self.offsets = [self.starting_offset + i *
                            self.spacing for i in range(self.point_count)]
            np.savetxt(self.offsets_file, self.offsets)

        self.point = 0
        self.label = None
        self.createWidgets()

    def createWidgets(self):
        f0 = tk.Frame()
        self.fig, self.ax = plt.subplots(1, 2, figsize=(12, 12))

        self.canvas = FigureCanvasTkAgg(self.fig, f0)
        self.canvas._tkcanvas.pack(fill=tk.BOTH, expand=1)

        f0.pack(fill=tk.BOTH, expand=1)

        self.update_fig()

    def restack_offsets(self, idx):
        """Reset the position of all points above idx using the mean spacing 
        of all points below idx"""
        spacing = np.diff(self.offsets)[:idx].mean()
        for i in range(idx + 1, len(self.offsets)):
            self.offsets[i] = self.offsets[i-1] + spacing

    def update_fig(self):
        for ax in self.ax:
            ax.clear()
        point, angle = microns_into_pattern(self.offsets[self.point],
                                            self.pattern, self.px_per_mm * 1e-3)
        display_patch(self.pic, point, angle, self.spacing, self.pitch,
                      self.pattern, axs=self.ax)
        self.label = tk.Label(
            self, text=f"point: {self.point}  {point}  {angle} deg")
        self.label.place(x=10, y=10)
        self.canvas.draw()

    def keypress(self, event):
        INC = 10
        if event.keysym == 'Right':
            if self.point < len(self.offsets):
                self.point += 1
        elif event.keysym == 'Left':
            if self.point > 0:
                self.point -= 1
        elif event.keysym == 'Up':
            self.offsets[self.point] += INC
        elif event.keysym == 'Down':
            self.offsets[self.point] -= INC
        elif event.char == 'r':
            print('restacking offsets')
            self.restack_offsets(self.point)
        elif event.char == 's':
            print(f'saving offests to {self.offsets_file}')
            np.savetxt(self.offsets_file, self.offsets)

        self.update_fig()


def run_patches_gui(im_path, pattern_path, offsets_file):
    app = Patches_GUI(im_path, pattern_path, offsets_file=offsets_file)
    app.bind("<Key>", app.keypress)
    app.mainloop()


if __name__ == "__main__":
    DIR = "E:\Dropbox\SPEED\Self Driving EHD\Data\Olympus mosaics"
    PIC = os.path.join(DIR, "10-mar-22__1.6V harmonics__10x.tif")
    PATTERN = os.path.join(DIR, "pattern.txt")

    offsets_file = os.path.join(DIR, "logs/offsetlist.txt")
    # LOG = os.path.join(DIR, "results 10 mar large noz 300 pitch.xlsx")

    run_patches_gui(PIC, PATTERN, offsets_file)
