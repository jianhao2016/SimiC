#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qizai <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
This is an implementation of the paper online NMF.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
import pickle
import time
import os
import argparse
import ipdb
import random
import copy
import itertools
import math
import numba as nb
from sklearn.linear_model import LassoLars, MultiTaskLasso
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import NMF
from sklearn.metrics import accuracy_score, adjusted_mutual_info_score
from .visualization import tsne_df
from .common_io import load_dataFrame, extract_df_columns, split_df_and_assignment
from .evaluation_metric import get_r_squared
from .sc_different_clustering import kmeans_clustering, nmf_clustering, spectral_clustering, evaluation_clustering
from .gene_id_to_name import load_dict, save_dict
from scipy.linalg import eigh, eigvalsh

def total_lasso(df_in, tf_list, target_list, lambda1):
    # df_in = pd.read_pickle(df_file_name)
    TF_mat = extract_df_columns(df_in, tf_list).values
    target_mat = extract_df_columns(df_in, target_list).values
    print('TF size:', TF_mat.shape)
    print('Target size:', target_mat.shape)

    multi_task_lasso_op = MultiTaskLasso(alpha = lambda1)
    multi_task_lasso_op.fit(TF_mat, target_mat)
    coef_multi_task_lasso = multi_task_lasso_op.coef_
    # print('L2 error: {:.4f}'.format(multi_task_lasso_op.score(TF_mat, target_mat)))
    
    m, n_x = TF_mat.shape
    assert m == target_mat.shape[0]

    # num_idpt_variable = min(n_x, m - 2)
    W_i = coef_multi_task_lasso
    W_i_avg = np.mean(W_i, axis = 1)
    W_avg_count = np.count_nonzero(W_i_avg > 1e-3)
    num_idpt_variable = min(m, W_avg_count) - 2

    Y_pred = TF_mat @ coef_multi_task_lasso.T
    sum_r2 = get_r_squared(target_mat, Y_pred, k = num_idpt_variable)

    print('L2 error: {:.4f}'.format(1/m * np.linalg.norm(Y_pred - 
        target_mat)**2))
    print('adj r2: {:.4f}'.format(sum_r2))
    return coef_multi_task_lasso


def find_top_k_TFs(top_k, query_gene_list, coef, tf_list, name_id_mapping):
    '''
    top_k: k largest coefficient
    query_gene_list: target genes to be queried
    coef: W.T in the regression model, coefficients of TFs with corresponding query_gene
    '''
    # top_k = 3
    # ipdb.set_trace()
    for co_tmp, tgene in zip(coef, query_gene_list):
        idx = np.argpartition(abs(co_tmp), -top_k)
        # print(idx)
        # importance_TF_list = tf_list[idx[-top_k:].astype(np.int)]
        importance_TF_list = []
        for i in idx[-top_k:]:
            gene_id = tf_list[int(i)]
            gene_name = name_id_mapping[gene_id]
            # importance_TF_list.append(tf_list[int(i)])
            importance_TF_list.append(gene_name)
        print('target genes:\n\t', name_id_mapping[tgene])
        print('number of non-zero coefficient: ', (abs(co_tmp) > 1e-2).sum())
        print('largest {} value of coefficient:'.format(top_k))
        print(co_tmp[idx[-top_k:]])
        print('TFs:\n\t', importance_TF_list)
        print('-' * 7)


def get_top_k_non_zero_targets(top_k, df, target_list):
    '''
    input: path to df file, top k choices, target list
    output: top k target genes in df
    '''
    # ipdb.set_trace()
    df_target = extract_df_columns(df, target_list)
    new_target_list = df_target.columns.values.tolist()
    sum_list = df_target.sum().values
    idx = np.argpartition(sum_list, -top_k)
    # print(idx)
    important_target_list = []
    for i in idx[-top_k:]:
        important_target_list.append(new_target_list[int(i)])
    # print(df[important_target_list].sum())
    return important_target_list


    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--numIter', type=int, default=1000)
    parser.add_argument('--method', type=str, choices=['kmeans', 'nmf', 'spectral'], 
            default = 'kmeans')
    parser.add_argument('--NF', type=float, default=1, help='normalization factor')

    parser.add_argument('--num_cells', type=int, default=500, 
            help='number of cells per cluster')
    parser.add_argument('--similarity', type=int, choices=[0, 1], default=1,
            help='0: no similarity constraint, 1: add similarity')
    parser.add_argument('--lambda1', type=float, default=1e-3,
            help='sparsity param in lasso')
    parser.add_argument('--lambda2', type=float, default=1e-5,
            help='param of similarity constraint')
    parser.add_argument('--data_type', type=str,
            choices=['raw', 'rwr', 'magic'], default = 'magic')

    args = parser.parse_args()

    # set number of iteration, lambda in lasso, epsilon in dictionary update and normalization factor
    print(args)
    numIter = args.numIter
    method = args.method
    _NF = args.NF
    expr_dtype = args.data_type
    similarity = (args.similarity == 1)
    cells_per_cluster = args.num_cells

    data_root = '/data/jianhao/clus_GRN'
    k_cluster = 3
    # raw_data = True
    set_of_gene = 'landmark'
    
    if expr_dtype == 'raw':
        p2df_file = os.path.join(data_root, 'raw_expression_data', 'pandas_dataframe_{}'.format(cells_per_cluster))
    elif expr_dtype == 'magic':
        file_name_df = 'magic_smoothed_all_genes_{}_cells_per_cluster'.format(cells_per_cluster)
        p2df_file = os.path.join(data_root, file_name_df)
    elif expr_dtype == 'rwr':
        file_name_df = 'smoothed_dataframe_{}_cells_per_cluster_all_gene_coexpression'.format(cells_per_cluster)
        p2df_file = os.path.join(data_root, 'ensembl94', file_name_df)




    #### BEGIN of the regression part
    # ipdb.set_trace()
    print('------ Begin the regression part...')
    df = pd.read_pickle(p2df_file)
    print('df in regression shape = ', df.shape)

    # p2tf_gene_list = os.path.join(data_root, 'diff_gene_list', 'gene_id_TF_and_ensembl_pickle')
    p2tf_gene_list = os.path.join(data_root, 'diff_gene_list', 'top_50_MAD_val_selected_TF_pickle')
    p2target_gene_list = os.path.join(data_root, 'diff_gene_list', 'gene_id_non_TF_and_ensembl_pickle')
    with open(p2tf_gene_list, 'rb') as f:
        tf_list = pickle.load(f)
        tf_list = list(tf_list)
    with open(p2target_gene_list, 'rb') as f:
        target_list = pickle.load(f)
        target_list = list(target_list)


    lambda1 = args.lambda1
    lambda2 = 0

    ############### end of cv ####################


    #### optimize using RCD
    # ipdb.set_trace()
    m, _ = df.shape
    randperm = np.random.permutation(m)
    train_idx = randperm[:-int(m * 0.2)]
    test_idx = randperm[-int(m*0.2):]
    df_train = df.loc[train_idx]
    df_test = df.loc[test_idx]

    query_target_list = get_top_k_non_zero_targets(10, df_train, target_list)
    print('-' * 7)

    print(' --- train set')
    W_trained = total_lasso(df_train, tf_list, query_target_list, lambda1)
    X_test = df_test[tf_list].values
    Y_test = df_test[query_target_list].values
    Y_pred = X_test @ W_trained.T

    m, n_x = X_test.shape
    assert m == Y_test.shape[0]
    W_i = W_trained
    W_i_avg = np.mean(W_i, axis = 1)
    W_avg_count = np.count_nonzero(W_i_avg > 1e-3)
    num_idpt_variable = min(m, W_avg_count) - 2
    # num_idpt_variable = min(n_x, m - 2)
    sum_r2 = get_r_squared(Y_test, Y_pred, k = num_idpt_variable)
    print(' --- test set')
    print('l2 loss: {:.4f}'.format(1/m * np.linalg.norm(Y_pred - Y_test)**2))
    print('adj r2: {:.4f}'.format(sum_r2))

    # # ipdb.set_trace()
    # # variable_list[1] *= 1.1
