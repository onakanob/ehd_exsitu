# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 11:40:41 2022

@author: Oliver
"""
import os
import json
import numpy as np

# FILE = 'params.json'
# testd = {'param1': 1,
#          'param2': '2'}
# with open("sample.json", "w") as outfile:
#     json.dump(dictionary, outfile)

# with open(FILE, 'w') as f:
#     json.dump(testd, f)

# with open(FILE, 'r') as f:
#     testd2 = json.load(f)

# print(testd)
# print(testd2)

DIR = "E:\Dropbox\SPEED\Self Driving EHD\Data\Olympus mosaics"
pattern_file = os.path.join(DIR, "logs/pattern_params.json")

px_per_mm = 795
dictionary = {'px_per_mm': px_per_mm,   # 10x Olympus: 795
              'scale': px_per_mm / 1e4,  # SIJ patterns use units 1e-4 mm
              'pattern_offset': [270, 310],
              'theta': np.radians(1.4)}  # degrees clockwise}

with open(pattern_file, 'w') as f:
    json.dump(dictionary, f)
