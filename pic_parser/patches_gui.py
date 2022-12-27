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

from .pattern_tools import (microns_into_pattern, align_pattern,
                            display_pattern, patchmaker)
from .patch_tools import (parse_patch, histogram_patches,
                          open_and_preprocess_pic)
from .param_tools import Params_Manager


class Image_Frame(tk.Frame):
    def __init__(self, fig, ax):
        super().__init__()
        self.fig = fig
        self.ax = ax

        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas._tkcanvas.pack(fill=tk.BOTH, expand=1)

    def draw(self, image):
        self.ax.clear()
        self.ax.imshow(image)
        self.canvas.draw()


class Props_Frame(tk.Frame):
    def __init__(self, restack_fun):
        ROWS = 9
        super().__init__()

        for r in range(ROWS):
            self.rowconfigure(r, weight=1)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        r = 0
        self.elements = {}

        tk.Label(self, text="Press <Enter> to save changes!").grid(row=r,
                                                                   column=0,
                                                                   sticky="w")
        r += 1
        tk.Label(self, text="Patch Index:  ").grid(row=r, column=0, sticky="e")
        self.elements['index'] = tk.IntVar()
        tk.Entry(self, textvariable=self.elements['index']).grid(row=r,
                                                                 column=1,
                                                                 sticky="w")
        r += 1
        tk.Label(self, text="Include Me:  ").grid(row=r, column=0, sticky="e")
        self.elements['include_me'] = tk.IntVar()
        tk.Checkbutton(self, variable=self.elements['include_me']).grid(row=r,
                                                                     column=1,
                                                                     sticky="w")
        r += 1
        tk.Label(self, text="De-Clog Event:  ").grid(row=r, column=0, sticky="e")
        self.elements['de_clog'] = tk.IntVar()
        tk.Checkbutton(self, variable=self.elements['de_clog']).grid(row=r,
                                                                     column=1,
                                                                     sticky="w")
        r += 1
        tk.Label(self, text="Offset:  ").grid(row=r, column=0, sticky="e")
        self.elements['offset'] = tk.StringVar(value="")
        tk.Entry(self, textvariable=self.elements['offset']).grid(row=r,
                                                                  column=1,
                                                                  sticky="w")
        r += 1
        tk.Label(self, text="image_thresh:  ").grid(row=r, column=0, sticky="e")
        self.elements['image_thresh'] = tk.StringVar(value="")
        tk.Entry(self, textvariable=self.elements['image_thresh']).grid(row=r,
                                                                  column=1,
                                                                  sticky="w")
        r += 1
        tk.Label(self, text="image_lowthresh:  ").grid(row=r, column=0,
                                                       sticky="e")
        self.elements['image_lowthresh'] = tk.StringVar(value="")
        tk.Entry(self, textvariable=self.elements['image_lowthresh']).grid(row=r,
                                                                     column=1,
                                                                     sticky="w")
        r += 1
        tk.Label(self, text="image_minsize:  ").grid(row=r, column=0, sticky="e")
        self.elements['image_minsize'] = tk.StringVar(value="")
        tk.Entry(self, textvariable=self.elements['image_minsize']).grid(row=r,
                                                                   column=1,
                                                                   sticky="w")
        r += 1
        tk.Label(self, text="positions:  ").grid(row=r, column=0, sticky="e")
        tk.Button(self, text="restack", command=restack_fun).grid(row=r,
                                                                  column=1,
                                                                  sticky="w")


    def set_vals(self, dictionary):
        index = dictionary.get('index')
        if index is not None:
            self.elements['index'].set(index)

        include_me = dictionary.get('include_me')
        if include_me is not None:
            if include_me:
                self.elements['include_me'].set(1)
            else:
                self.elements['include_me'].set(0)

        declog = dictionary.get('de_clog')
        if declog is not None:
            if declog:
                self.elements['de_clog'].set(1)
            else:
                self.elements['de_clog'].set(0)

        offset = dictionary.get('offset')
        if offset is not None:
            self.elements['offset'].set(offset)

        image_thresh = dictionary.get('image_thresh')
        if image_thresh is not None:
            self.elements['image_thresh'].set(image_thresh)

        image_lowthresh = dictionary.get('image_lowthresh')
        if image_lowthresh is not None:
            self.elements['image_lowthresh'].set(image_lowthresh)

        image_minsize = dictionary.get('image_minsize')
        if image_minsize is not None:
            self.elements['image_minsize'].set(image_minsize)


    def get_vals(self):
        if self.elements['include_me'].get():
            include_me = True
        else:
            include_me = False
        if self.elements['de_clog'].get():
            de_clog = True
        else:
            de_clog = False
        return {'index':           int(self.elements['index'].get()),
                'include_me':      include_me,
                'de_clog':         de_clog,
                'offset':          int(self.elements['offset'].get()),
                'image_thresh':    int(self.elements['image_thresh'].get()),
                'image_lowthresh': int(self.elements['image_lowthresh'].get()),
                'image_minsize':   int(self.elements['image_minsize'].get())}


class Micros_GUI(Params_Manager, tk.Tk):
    def __init__(self, params_file, cold_start=False):
        Params_Manager.__init__(self, params_file, cold_start=cold_start)
        tk.Tk.__init__(self)

        # Static:
        self.pic = open_and_preprocess_pic(self.pic_path())
        self.pattern = align_pattern(csv=self.pattern_path(),
                                     scale=self.params['px_per_mm'] / 1e4,
                                     theta=self.params['theta'],
                                     offset=self.params['pattern_offset'])
        self.geometry("1200x900")
        self.title("Micrograph GUI")


class Patches_GUI(Micros_GUI):
    # def __init__(self, im_path, pattern_path, params_file):
    def __init__(self, params_file):
        super().__init__(params_file)

        self.title("Patches Selector")

        # State:
        self.loc = 0

        # Initialize:
        self.rowconfigure(0, weight=1)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)

        self.pattern_fig, self.pattern_ax = plt.subplots(figsize=(6, 12))
        self.patch_fig, self.patch_ax = plt.subplots(figsize=(6, 12))

        self.pattern_frame = Image_Frame(self.pattern_fig, self.pattern_ax)
        self.pattern_frame.grid(row=0, column=0, sticky="nsew")

        self.patch_frame = Image_Frame(self.patch_fig, self.patch_ax)
        self.patch_frame.grid(row=0, column=1, sticky="nsew")

        self.inputs_frame = Props_Frame(restack_fun=lambda:
                                        self.restack_offsets(self.loc))
        self.inputs_frame.grid(row=0, column=2, sticky="w")

        self.bind("<Key>", self.keypress)
        self.update_location(0)


    def update_location(self, new_loc):
        if new_loc < 0:
            raise ValueError('Patch index out of range: negative.')
        if new_loc >= self.params['point_count']:
            raise ValueError('Patch index out of range: too large.')

        self.loc = new_loc
        self.render_frames()
        self.update_input_frame()


    def render_frames(self):
        point, angle = microns_into_pattern(self.params[self.loc]['offset'],
                                            self.pattern,
                                            self.params['px_per_mm'] * 1e-3)
        self.render_patch(point, angle)
        self.render_pattern(point)


    def render_patch(self, point, angle):
        patch = patchmaker(self.pic,
                           self.params['spacing'] * self.params['px_per_mm']\
                               * 1e-3 * 1.2,
                           self.params['pitch'] * self.params['px_per_mm']\
                               * 1e-3 * 1.5,
                           int(point[0]), int(point[1]),
                           angle)
        # import ipdb; ipdb.set_trace()

        _, _, _, threshold, low_thresh, min_size = self.get_patch_params()
        _, contours = parse_patch(patch.copy(),
                            threshold=threshold,
                            low_thresh=low_thresh,
                            min_size=min_size,
                            um_per_px=1e3 / self.params['px_per_mm'],
                            return_image=True)
        self.patch_frame.draw(contours)


    def render_pattern(self, point):
        """Draw the overall pattern and patch location on to the current
        pattern. This does not use the Frame.draw() method because
        display_pattern makes multiple sequential changes to a live pyplot axis
        object."""
        self.pattern_ax.clear()
        display_pattern(self.pic,
                        self.pattern,
                        point=point,
                        axs=self.pattern_ax)
        self.pattern_frame.canvas.draw()


    def update_input_frame(self):
        include_me, offset, de_clog, image_thresh,\
            image_lowthresh, image_minsize = self.get_patch_params()
    
        self.inputs_frame.set_vals({'index': self.loc,
                                    'include_me': include_me,
                                    'offset': offset,
                                    'de_clog': de_clog,
                                    'image_thresh': image_thresh,
                                    'image_lowthresh': image_lowthresh,
                                    'image_minsize': image_minsize})


    def get_patch_params(self, loc=None):
        if loc is None:
            loc = self.loc

        include_me = self.params[loc]['include_me']
        offset = self.params[loc]['offset']
        de_clog = self.params[loc]['de_clog']

        threshold = self.params[loc]['image_thresh']
        if threshold is None:
            threshold = self.params['image_thresh']

        low_thresh = self.params[loc]['image_lowthresh']
        if low_thresh is None:
            low_thresh = self.params.get('image_lowthresh')

        min_size = self.params[loc]['image_minsize']
        if min_size is None:
            min_size = self.params['image_minsize']

        return include_me, offset, de_clog, threshold, low_thresh, min_size


    def set_patch_params(self, new_params, loc=None):
        if loc is None:
            loc = self.loc

        rewrite = False
        for key in self.DEFAULT_DICT.keys():
            val = new_params.get(key)
            if (val is not None):
                if val == self.params.get(key):
                    new_val = None  # Use default value
                else:
                    new_val = val

                if new_val != self.params[loc][key]:
                    self.params[loc][key] = new_val
                    rewrite = True

        if rewrite:
            self.write_paramfile()


    def keypress(self, event):
        if event.keysym == "Return":
            self.press_return()
        elif event.keysym == 'Prior':
            if self.loc < self.params['point_count'] - 1:
                self.update_location(self.loc + 1)
        elif event.keysym == 'Next':
            if self.loc > 0:
                self.update_location(self.loc - 1)
        else:
            return


    def press_return(self):
        vals = self.inputs_frame.get_vals()  # Get current values from GUI
        self.set_patch_params(vals)          # Update params in dict and JSON
        self.update_location(vals['index'])
    

    def restack_offsets(self, idx):
        offsets = self.stack_offsets(idx=idx)
        j = 0
        for i in range(idx, self.params['point_count']):
            self.params[i]['offset'] = int(offsets[j])
            j += 1
        self.write_paramfile()


def run_patches_gui(params_file, test=False):
    app = Patches_GUI(params_file)
    if test:                    # return the GUI object for unit tests
        return app
    app.mainloop()              # Run the GUI


class Align_GUI(Micros_GUI):
    docstring="""Make changes to pattern_params to overlay the pattern line
    and first feature over the image. The stars denote: first point, then every
    two experiments, and the final point. Click on the window to refresh
    view. Press 'r' to repopulate all point offsets in the params file."""

    def __init__(self, params_file, cold_start=False):
        super().__init__(params_file, cold_start=cold_start)

        histogram_patches({'pic': self.pic}, bins=128, xlim=(90, 256),
                          output=os.path.join(os.path.dirname(params_file),
                                              'brightness_histogram.png'))

        self.title("Pattern Aligner")
        self.create_widgets()
        self.bind("<Button-1>", self.leftclick)
        self.bind("<r>", self.restack)
        print(self.docstring)

    def create_widgets(self):
        self.pattern_fig, self.pattern_ax = plt.subplots(figsize=(12, 12))
        self.pattern_frame = Image_Frame(self.pattern_fig, self.pattern_ax)
        self.pattern_frame.pack(fill=tk.BOTH, expand=True)
        self.update_fig()

    def update_fig(self):
        DISPLAY_EVERY = 2
        self.load_paramfile()
        self.pattern = align_pattern(csv=self.pattern_path(),
                                     scale=self.params['px_per_mm'] / 1e4,
                                     theta=self.params['theta'],
                                     offset=self.params['pattern_offset'])

        # Update offsets - this does not change the JSON file yet
        offsets = self.stack_offsets()
        # Grab some points to display
        points = []
        for o in np.concatenate((offsets[0::DISPLAY_EVERY], [offsets[-1]])):
            point, _ = microns_into_pattern(o,
                                            self.pattern,
                                            self.params['px_per_mm'] * 1e-3)
            points.append(point)

        # Render the new pattern and point positions
        self.pattern_ax.clear()
        display_pattern(self.pic,
                        self.pattern,
                        point=points,
                        axs=self.pattern_ax)
        self.pattern_frame.canvas.draw()

    def leftclick(self, event):
        self.update_fig()

    def restack(self, event):
        self.initialize_params(restack_offsets=True)


def run_alignment_gui(params_file, test=False,
                      cold_start=False):
    app = Align_GUI(params_file, cold_start=cold_start)
    if test:
        return app
    app.mainloop()
