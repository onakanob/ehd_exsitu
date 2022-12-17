# -*- coding: utf-8 -*-
"""


Created on 15 December 2022

@author: Oliver Nakano-Baker
"""
import json

import numpy as np


def keys_to_ints(dictionary):
    d = {}
    for k, v in dictionary.items():
        try:
            d[int(k)] = v
        except:
            d[k] = v
    return d


class Params_Manager():
    DEFAULT_DICT = {'include_me': True,
                    'offset': None,
                    'image_thresh': None,
                    'image_lowthresh': None,
                    'image_minsize': None}

    def __init__(self, paramfile):
        super().__init__()
        self.paramfile = paramfile
        self.params = None
        self.load_paramfile()
        self.initialize_params()

    def load_paramfile(self):
        with open(self.paramfile, 'r') as f:
            self.params = json.load(f, object_hook=keys_to_ints)

    def write_paramfile(self):
        with open(self.paramfile, 'w', encoding="utf-8") as f:
            json.dump(self.params, f,
                      ensure_ascii=False,
                      indent=4)
            print(f'Wrote to {self.paramfile}')


    def stack_offsets(self, idx=0, spacing=None):
        """Calculate the position of points above idx using the specified
        spacing."""             # TODO this doesn't work
        offsets = [self.params[idx]['offset']]
        if spacing is None:
            spacing = self.params['spacing']
        for _ in range(idx+1, self.params['point_count']):
            offsets.append(spacing + offsets[-1])
        # offsets = np.arange(idx, self.params['point_count']) * spacing\
        #     + self.params['starting_offset']
        return np.array(offsets).astype(int)


    def initialize_params(self, restack_offsets=False):
        offsets = self.stack_offsets()
        for i in range(self.params['point_count']):
            if self.params.get(i) is None:
                self.params[i] = self.DEFAULT_DICT.copy()
            if (self.params[i]['offset'] is None) or (restack_offsets):
                self.params[i]['offset'] = int(offsets[i])
        self.write_paramfile()
