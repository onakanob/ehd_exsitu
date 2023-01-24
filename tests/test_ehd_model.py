# -*- coding: utf-8 -*-
"""
Test methods for ehd_models.model etc.

TODO: not all model types are currently covered by this test script.

Created January 14, 2023

@author: Oliver Nakano-Baker
"""

import os
import sys

import pytest
import numpy as np
import pandas as pd

sys.path.append('..')
from ehd_models import EHD_Model
from ehd_dataset import EHD_Loader


XTYPE = "vector"  # vector, normed_squares, v_normed_squares
YTYPE = "jetted"            # jetted, max_width
FILTERS = [
           ('Wavegen', lambda x: x, 'harmonics'),
           ('clogging', lambda x: x, False)
          ]


def test_models():
    INDEX = './dataset_index.xlsx'
    SAVE_TO = './test_models/'
    loader = EHD_Loader(index_file=INDEX)

    architectures = [
        "reweight_RF",
        "only_pretrained_RF_class",
        "MLE_class",
        "MLE",
        "cold_RF_class",
        "v_normed_Ridge",
        "v_normed_MLP",
        "warm_MLP",
        "warm_RF",
    ]
    _, eval_set, eval_name =\
        loader.folded_dataset(fold=0,
                              xtype=XTYPE,
                              ytype=YTYPE,
                              pretrain=True,
                              filters=FILTERS)
    for a in architectures:
        model = EHD_Model(architecture=a,
                          params={})
        # Only one dataset in the test bank, so we just get the eval set back:

        model.pretrain(eval_set)
        model.retrain(eval_set)
        output = model.predict(eval_set['X'])
        assert isinstance(model.evaluate(eval_set, train_sizes=[20]),
                          pd.DataFrame)

        if not os.path.isdir(SAVE_TO):
            os.mkdir(SAVE_TO)
        outpath = os.path.join(SAVE_TO, a + ".pickle")

        model.save(outpath)     # Save it
        model = None            # Nuke it
        model = EHD_Model.load(outpath)  # Reload it
        assert all(output == model.predict(eval_set['X']))


if __name__=="__main__":
    test_models()
