#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qizai <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
This file can be ran on server.
require T1, T2, and gene list already exist.
"""

import numba as nb
import numpy as np
import pandas as pd
import pickle
import ipdb
import os
import time
from numba import cuda, float64
from common_io import load_dataFrame

@nb.jit(nopython=True)
def fast_RWR_on_GGN(T1, T2, c, e_i):
    '''
    T1, T2: matrice inverion from pre_computation_step.
    c: restart probability.
        r_i = c* W * r_i + (1-c) e_i
    e_i: node i in query
    '''
    r_i = (1 - c) * ( T1 @ e_i + c * T2 @ e_i)
    return r_i

def get_smooth_mat(T1, T2, new_gene_list, c_rwr):

    len_of_gene = len(new_gene_list)
    smooth_mat_V = np.zeros((len_of_gene, len_of_gene))
    e_i = np.zeros(len_of_gene)

    t_s = time.time()
    print('start looping smooth matrix')
    for idx in range(len_of_gene):
        if idx % 1000 == 1:
            t_e = time.time()
            print('{} iteration cost time {:.4f}'.format(idx, t_e - t_s))

        e_i[idx] = 1
        r_i = fast_RWR_on_GGN(T1, T2, c_rwr, e_i) 
        smooth_mat_V[:, idx] = r_i
        e_i = np.zeros(len_of_gene)

    return smooth_mat_V



if __name__ == '__main__':
    root_path = '/data/jianhao/scRNA_seq/'
    # df_file = 'pandas_dataframe'
    # feature_file = 'df_feature_column'
    # feature_file = 'ensembl_gene_list'
    # p2df = os.path.join(root_path, df_file)

    part_k = 5
    c_rwr = 0.2

    # T1_file = os.path.join(root_path, 'fast_rwr_T1.npy')
    # T2_file = os.path.join(root_path, 'fast_rwr_T2.npy')
    gene_file = os.path.join(root_path, 'imputation_data', 'fast_rwr_gene_list')
    V_file = os.path.join(root_path, 'imputation_data', 'smooth_matrix_V.npy')
    X_df = os.path.join(root_path, 'pandas_dataframe')

    # T1 = np.load(T1_file)
    # T2 = np.load(T2_file)
    with open(gene_file, 'rb') as f:
        new_gene_list = pickle.load(f)


    #smooth_mat_V = get_smooth_mat(T1, T2, new_gene_list, c_rwr)
    #smooth_file = os.path.join(root_path, 'smooth_matrix_V')
    #np.save(smooth_file, smooth_mat_V)

    smooth_matrix_V = np.load(V_file)
    X, Y, feature_cols = load_dataFrame(X_df, new_gene_list)

    # X, Y, feature_cols = load_dataFrame(p2df, new_gene_list)
    print('start computing!')
    smoothed_X = X @ smooth_matrix_V
    print('done!')

    df_smoothed_X = pd.DataFrame(data=smoothed_X, columns = feature_cols)
    smoothed_gene_file = os.path.join(root_path, 'gene_in_smoothed_X')
    with open(smoothed_gene_file, 'wb') as f:
        pickle.dump(feature_cols, f)

    df_smoothed_X['label'] = Y
    smoothed_X_file = os.path.join(root_path, 'smoothed_dataframe_100_cells_per_cluster')
    df_smoothed_X.to_pickle(smoothed_X_file)

