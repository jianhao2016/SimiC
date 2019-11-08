#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qizai <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
This script is to load the 10x Genomics data
"""

import csv
import os
import scipy.io
import numpy as np
import pandas as pd
import pickle
from scipy.sparse import coo_matrix, hstack

def load_mouse_data(p2data):
    data_file = '/data/jianhao/clus_GRN/mouse_GSE_dataset/GSE60361_C1-3005-Expression.txt'
    df = pd.read_table(data_file, index_col = 0)
    df = df.transpose()
    gene_list = list(df.columns.values)
    
    with open('/data/jianhao/clus_GRN/diff_gene_list/df_feature_column_mouse', 'wb') as f:
        pickle.dump(gene_list, f)
    
    label_file = '/data/jianhao/clus_GRN/mouse_GSE_dataset/GSE60361.annotations.csv'
    cell_type_list = pd.read_csv(label_file, index_col = 0)
    
    df['label'] = cell_type_list
    print('Size of data frame', df.shape)
    print(df['label'].shape)
    df.to_pickle('/data/jianhao/clus_GRN/raw_expression_data/pandas_dataframe_mouse')
