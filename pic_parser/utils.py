import numpy as np


def cell_to_array(string):
    split_string = string[1:-2].split()
    return np.array(split_string).astype(float)