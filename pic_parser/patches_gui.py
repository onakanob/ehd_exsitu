# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 00:46:17 2022

@author: Oliver
"""
import os
import json

import numpy as np
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image
from PIL.ImageOps import grayscale

from .pattern_tools import (microns_into_pattern, align_pattern,
                            display_pattern, display_patch)


class Micros_GUI(tk.Tk):
    # px_per_mm = 795  # 10x Olympus: 10 Mar '22
    # px_per_mm = 975/2  # 20x Olympus: 29 Mar '22

    # scale = px_per_mm / 1e4  # SIJ patterns use units 1e-4 mm
    # pattern_offset = [270, 310]
    # theta = np.radians(1.4)  # degrees clockwise

    # point_count = 112

    # # microns:
    # pitch = 300
    # spacing = 500
    # starting_offset = 16.52e3

    def __init__(self, paramfile):
        super().__init__()
        self.paramfile = paramfile
        self.load_paramfile()

    def load_paramfile(self):
        with open(self.paramfile, 'r') as f:
            params = json.load(f)
        self.px_per_mm = params['px_per_mm']
        self.scale = self.px_per_mm / 1e4
        self.pattern_offset = params['pattern_offset']
        self.theta = params['theta']
        self.spacing = params['spacing']
        self.pitch = params['pitch']
        self.starting_offset = params['starting_offset']
        self.point_count = params['point_count']


class Align_GUI(Micros_GUI):  # TODO
    docstring = """Make changes to pattern_params to overlay the pattern line
    and first feature over the image. Click on the window to refresh view."""

    def __init__(self, picture, pattern_file, paramfile):
        super().__init__(paramfile)

        self.pic = Image.open(picture)
        self.pic = np.fliplr(np.flipud(np.array(grayscale(self.pic)).T))
        self.pattern_file = pattern_file
        self.pattern = align_pattern(self.pattern_file, self.scale, self.theta,
                                     self.pattern_offset)

        self.title("Pattern Aligner")
        self.geometry("1200x900")

        # self.label = None
        self.createWidgets()
        print(self.docstring)

    def createWidgets(self):
        f0 = tk.Frame()
        self.fig = plt.figure(figsize=(16, 16))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, f0)
        self.canvas._tkcanvas.pack(fill=tk.BOTH, expand=1)
        f0.pack(fill=tk.BOTH, expand=1)

        self.update_fig

    def update_fig(self):
        self.load_paramfile()
        self.pattern = align_pattern(self.pattern_file, self.scale, self.theta,
                                     self.pattern_offset)
        self.ax.clear()
        
        point, _ = microns_into_pattern(self.starting_offset,
                                        self.pattern, self.px_per_mm * 1e-3)
        display_pattern(self.pic, self.pattern, point=point, axs=self.ax)
        # self.label = tk.Label(
        #     self, text=f"point: {self.point}  {point}  {angle} deg")
        # self.label.place(x=10, y=10)
        self.canvas.draw()

    def leftclick(self, event):
        self.update_fig()


class Patches_GUI(Micros_GUI):
    docstring = """controls: left/right to move between positions. up/down to
    adjust position forward or back. r==restack all future positions. s==save
    current positions. g==go to position; type an int in terminal"""
    
    def __init__(self, picture, pattern_file, paramfile,
                 offsets_file='offsetlist.txt'):
        super().__init__(paramfile)

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
        print(self.docstring)

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
        INC = 25
        if event.keysym == 'Right':
            if self.point < len(self.offsets):
                self.point += 1
        elif event.keysym == 'Left':
            if self.point > 0:
                self.point -= 1
        elif event.keysym == 'g':
            self.point = int(input("Go to position: "))
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


def run_patches_gui(im_path, pattern_path, params_file, offsets_file):
    app = Patches_GUI(im_path, pattern_path, params_file, offsets_file=offsets_file)
    app.bind("<Key>", app.keypress)
    app.mainloop()


def run_alignment_gui(im_path, pattern_path, params_file):
    app = Align_GUI(im_path, pattern_path, params_file)
    app.bind("<Button-1>", app.leftclick)
    app.mainloop()

if __name__ == "__main__":
    DIR = "E:/Dropbox/SPEED/Self Driving EHD/Data/Olympus mosaics"
    DIR = "E:/Dropbox/SPEED/Self Driving EHD/Data/29-Mar-2022 lg 1cm 300 points"
    PIC = os.path.join(DIR, "10-mar-22__1.6V harmonics__10x.tif")
    PATTERN = os.path.join(DIR, "pattern.txt")

    params_file = os.path.join(DIR, "logs/pattern_params.json")
    offsets_file = os.path.join(DIR, "logs/offsetlist.txt")
    # LOG = os.path.join(DIR, "results 10 mar large noz 300 pitch.xlsx")

    # run_patches_gui(PIC, PATTERN, offsets_file, params_file)
    app = Patches_GUI(im_path, pattern_path, params_file,
                      offsets_file=offsets_file)
