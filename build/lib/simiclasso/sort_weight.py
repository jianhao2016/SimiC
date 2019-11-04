#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 jianhao2 <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
This script take saved dictionary after clus_GRN as input
     = {
        'weight_dic': trained_weight_dict,
        'TF_ids' : gene names of TFs,
        'query_targets': target genes being queried.
"""

import numpy as np
import csv
import os
import pandas as pd
import argparse
from .gene_id_to_name import load_dict, save_dict

def convert_weights_to_count(W_i, top_k_weights):
    '''
    convert a raw real value weight matrix, to binary matrix,
    top_k_weights in each column of W_i (corresponding to one vector)
    will 1, 
    other will be 0
    '''
    count_mat = np.zeros_like(W_i)
    tmp_mat = np.argpartition(abs(W_i), -top_k_weights, axis = 0)
    top_k_idx_mat = tmp_mat[-top_k_weights:, :]
    for col_idx in range(W_i.shape[1]):
        row_idx = top_k_idx_mat[:, col_idx]
        count_mat[row_idx, col_idx] = 1
    return count_mat

def get_top_k_pairs_from_mat(mat, top_k_pairs):
    '''
    sort the input mat in elements
    return the index in pair (row_idx, col_idx) 
    of top k pairs
    '''
    flat_order = np.argsort(mat, axis = None)
    idx_pair_all = np.unravel_index(flat_order, mat.shape)
    if top_k_pairs != None:
        idx_top_k_pairs = tuple(arr[-1:-1-top_k_pairs:-1] for arr in idx_pair_all)
    else:
        # get all sorted pairs idx
        idx_top_k_pairs = tuple(arr[::-1] for arr in idx_pair_all)
    return idx_top_k_pairs

def get_gene_link_from_pairs_of_idxs(TF_list, target_list, 
        val_mat, idx_top_k_pairs):
    str_2_saved = 'TF,Target gene,value'
    df_2_saved = pd.DataFrame(data = None, columns = str_2_saved.split(','))
    k = 1
    for tf_idx, tg_idx in zip(*idx_top_k_pairs):
        new_item = '{},{},{:.4f}'.format(
                TF_list[tf_idx],
                target_list[tg_idx], 
                val_mat[tf_idx, tg_idx])
        df_2_saved.loc[k] = new_item.split(',')
        k += 1
        str_2_saved += '\n' + new_item
    return str_2_saved, df_2_saved

def main_sort(p2saved_dict, top_k):
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_cells', type=int, default=100, 
            help='number of cells per cluster')
    parser.add_argument('--similarity', type=int, choices=[0, 1],
            help='0: no similarity constraint, 1: add similarity')
    args = parser.parse_args()

    num_cells = args.num_cells
    similarity = (args.similarity == 1)
    k_cluster = 3
    root_path = '/home/jianhao2/clus_GRN/code'
    file_name = '{}_cells_{}_cluster_{}_similarity'.format(
            num_cells, k_cluster, similarity)

    p2saved_dict = os.path.join(root_path, '/results_jan_29/lambda1_3_lambda2_3', file_name)
    full_dict = load_dict(p2saved_dict)
    print(full_dict.keys())
    # print(full_dict['weight_dic'])
    trained_weight_dict = full_dict['weight_dic']
    TF_list = full_dict['TF_ids']
    target_list = full_dict['query_targets']
    # print(type(TF_list))
    # print(TF_list[:10])

    n_x = len(TF_list)
    n_y = len(target_list)

    # count_mat and sum_mat are matirx in the form of:
    #         | target 1 | target 2 | target 3
    # TF 1    |    xx    |   xx     |     xx
    # TF 2    |    xx    |   xx     |     xx
    # then we can sort the matrix, and get top k of the pairs
    count_mat = np.zeros((n_x, n_y))
    sum_mat = np.zeros((n_x, n_y))
    top_k_weights = 20
    for label in trained_weight_dict:
        W_i = trained_weight_dict[label]
        assert (n_x, n_y) == W_i.shape
        sum_mat += abs(W_i)
        count_mat += convert_weights_to_count(W_i, top_k_weights)

    top_k_pairs = None
    idx_top_k_pairs = get_top_k_pairs_from_mat(sum_mat, top_k_pairs)
    tmp_str, tmp_df = get_gene_link_from_pairs_of_idxs(TF_list,
            target_list, sum_mat, idx_top_k_pairs)
    # print(tmp_str)
    print(tmp_df[:10])
    file_name = os.path.join(root_path, 'gene_links2',
            '{}_cells_{}_similarity_sum_order.csv'.format(num_cells,
                similarity))
    tmp_df.to_csv(file_name, sep='\t', encoding='utf-8')

    top_k_pairs = None
    idx_top_k_pairs = get_top_k_pairs_from_mat(count_mat, top_k_pairs)
    tmp_str, tmp_df = get_gene_link_from_pairs_of_idxs(TF_list,
            target_list, count_mat, idx_top_k_pairs)
    # print(tmp_str)
    print(tmp_df[:10])
    file_name = os.path.join(root_path, 'gene_links2',
            '{}_cells_{}_similarity_count_order.csv'.format(num_cells,
                similarity))
    tmp_df.to_csv(file_name, sep='\t', encoding='utf-8')



