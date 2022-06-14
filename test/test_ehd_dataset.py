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
        self.assertTrue(loader is not None)
        df = loader.get_datasets()[0]
        corr, _ = pearsonr(df.area,
                           df.wave.apply(lambda x: np.sum(np.abs(x))))
        self.assertTrue(corr > 0.5)
        corr, _ = pearsonr(df.area,
                           df.vector.apply(lambda x: np.sqrt(np.sum(x**2))))
        self.assertTrue(corr > 0.7)



if __name__ == '__main__':
    unittest.main()
