# -*- coding: utf-8 -*-
"""
Test methods for ehd_dataset.dataset.

Created on June 11 2022

@author: Oliver Nakano-Baker
"""

import unittest
import sys

import numpy as np
from scipy.stats import pearsonr

sys.path.append('..')
from ehd_dataset.dataset import EHD_Loader

class TestDataset(unittest.TestCase):
    def testImport(self):
        INDEX = './dataset_index.xlsx'
        loader = EHD_Loader(index_file=INDEX)
        # This dataset should have index offset == 4
        self.assertTrue(len(loader.datasets) == 1)
        df = loader.get_datasets()[0]
        # Area has >0.5 corr vs absolute area under waveform
        corr, _ = pearsonr(df.area,
                           df.wave.apply(lambda x: np.sum(np.abs(x))))
        self.assertTrue(corr > 0.5)
        # Area has >0.7 corr vs L2-norm of the wavevector
        corr, _ = pearsonr(df.area,
                           df.vector.apply(lambda x: np.sqrt(np.sum(x**2))))
        self.assertTrue(corr > 0.7)


class TestModels(unittest.TestCase):
    pass


if __name__ == '__main__':
    unittest.main()
