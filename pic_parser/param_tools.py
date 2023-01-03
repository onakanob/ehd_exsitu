# -*- coding: utf-8 -*-
"""


Created on 15 December 2022

@author: Oliver Nakano-Baker
"""
import os, json

import numpy as np


def keys_to_ints(dictionary):
    d = {}
    for k, v in dictionary.items():
        try:
            d[int(k)] = v
        except:
            d[k] = v
    return d


class Params_Reader():
    def __init__(self, paramfile):
        self.paramfile = paramfile
        self.params = None
        self.load_paramfile()

    def load_paramfile(self):
        with open(self.paramfile, 'r') as f:
            self.params = json.load(f, object_hook=keys_to_ints)

    def pic_path(self):
        return os.path.join(os.path.dirname(self.paramfile),
                            self.params['picture'])

    def pattern_path(self):
        return os.path.join(os.path.dirname(self.paramfile),
                            self.params['pattern'])

    def retrieve(self, i, key):
        """Return the key associated with patch i. If that key is None in the
        dictionary, return the default value."""
        value = self.params[i].get(key)
        if value is None:
            return self.params.get(key)
        return value


class Params_Manager(Params_Reader):
    DEFAULT_DICT = {'include_me': True,
                    'de_clog': False,
                    'offset': None,
                    'image_thresh': None,
                    'image_lowthresh': None,
                    'image_minsize': None}

    def __init__(self, paramfile, cold_start=False):
        super().__init__(paramfile)
        self.initialize_params(cold_start=cold_start)

    def write_paramfile(self):
        with open(self.paramfile, 'w', encoding="utf-8") as f:
            json.dump(self.params, f,
                      ensure_ascii=False,
                      indent=4)
            print(f'Wrote to {self.paramfile}')

    def stack_offsets(self, idx=0, spacing=None):
        """Calculate the position of points above idx using the specified
        spacing."""             # TODO this doesn't work
        if idx == 0:
            offsets = [self.params['starting_offset']]
        else:
            offsets = [self.params.get(idx).get('offset')]
        if spacing is None:
            spacing = self.params['spacing']
        for _ in range(idx+1, self.params['point_count']):
            offsets.append(spacing + offsets[-1])
        # offsets = np.arange(idx, self.params['point_count']) * spacing\
        #     + self.params['starting_offset']
        return np.array(offsets).astype(int)

    def initialize_params(self, restack_offsets=False, cold_start=False):
        offsets = self.stack_offsets()
        for i in range(self.params['point_count']):
            if (self.params.get(i) is None) or cold_start:
                self.params[i] = self.DEFAULT_DICT.copy()
            if (self.params[i]['offset'] is None) or restack_offsets:
                self.params[i]['offset'] = int(offsets[i])
        self.write_paramfile()
