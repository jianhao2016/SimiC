#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qizai <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
This script is to load the data from GSE
"""

import csv
import os
import scipy.io
import numpy as np
import pandas as pd
import pickle
from scipy.sparse import coo_matrix, hstack

def load_GSE_data(p2data, p2cell_type = None, df_with_label = False):
    if os.path.isfile(p2data):
        data_file = p2data
    else:
        raise ValueError('{} is not a valid file'.format(p2data))
    df = pd.read_table(data_file, index_col = 0)
    df = df.transpose()
    gene_list = list(df.columns.values)
    
    # dense_mat = hstack(shrinked_mat_list).todense()
    # print('size of stack dense mat: ', dense_mat.shape)
    # np.save('dense_data', dense_mat)
    # X = dense_mat.T
    # df = pd.DataFrame(data = X, columns = gene_ids_tmp)
    # with open('/data/jianhao/clus_GRN/diff_gene_list/df_feature_column_human_MGH', 'wb') as f:
    #     pickle.dump(gene_list, f)
    
    cell_type_list = [a.split('_')[0] for a in df.index.values]
    
    df['label'] = cell_type_list
    wrong_ids = [a for a in df.index.values if a[:3] != 'MGH']
    df = df.drop(wrong_ids)
    
    print('Size of data frame', df.shape)
    print(df['label'].shape)
    df.to_pickle('/data/jianhao/clus_GRN/raw_expression_data/pandas_dataframe_human_MGH')
