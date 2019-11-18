#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 jianhao2 <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
This script will run the AUC mat calculation.
"""
import pickle
import numpy as np
import pandas as pd
import ipdb
import time

def normalized_by_target_norm(p2df, p2res):
    '''
    normalized the weight matrix w.r.t. target expression
    cut at the len of TFs, removed the bias term.
    '''
    with open(p2res, 'rb') as f:
        res_dict = pickle.load(f)

    weight_dic = res_dict['weight_dic']
    TF_ids = res_dict['TF_ids']
    target_ids = res_dict['query_targets']

    original_df = pd.read_pickle(p2df)
    original_df.columns = map(str.upper, original_df.columns)

    target_df = original_df[target_ids]
    target_norms = np.linalg.norm(target_df, axis = 0)

    normalized_weights = {}
    for label in weight_dic:
        normalized_mat = weight_dic[label] / target_norms
        normalized_weights[label] = normalized_mat[:len(TF_ids), :]

    return normalized_weights, original_df, TF_ids, target_ids


def cal_AUC_df(row_series_in, weight_series_in):
    df_new = pd.concat([row_series_in, row_series_in], axis = 1).T
    df_new = df_new.reset_index(drop = True)
    df_new.loc[1] = weight_series_in
    df_new.fillna(0)
#     ipdb.set_trace()
    df_sorted_by_row = df_new.sort_values(axis = 1, by = 0, ascending = False)
    len_of_genes = len(weight_series_in)
    running_sum = [np.sum(df_sorted_by_row.loc[1].values[:x+1]) for x in range(len_of_genes)]
    AUC_score = np.sum(running_sum) / (np.sum(weight_series_in) * len_of_genes)
    return AUC_score

def cal_AUC(row_vec_in, weight_vec_in):
    row_np = np.array(row_vec_in)
    weight_np = np.array(weight_vec_in)
    expression_descending_order = np.argsort(row_np)[::-1]
    ordered_weighted = weight_np[expression_descending_order]

    div_factor = 1
    len_of_genes = len(ordered_weighted) // div_factor
    sum_of_weight = np.sum(ordered_weighted[:len_of_genes])

    running_sum = [np.sum(ordered_weighted[:x+1]) for x in range(len_of_genes)]
    AUC_score = np.sum(running_sum) / (sum_of_weight * len_of_genes)
    return AUC_score


def get_AUCell_mat(original_df, weight_dict, TF_ids, target_ids, threshold = None):
    '''
    weight_mat: # num_TFs * num_Targets.
    target_ids: all targets, thresholding should only change the value to 0, not removing.
    '''
    df_in = original_df.reset_index(drop = True)
    original_index = original_df.index
    AUC_dict = {}
    for label in weight_dict:
        weight_mat = np.abs(weight_dict[label])
        AUC_mat = np.zeros([df_in.shape[0], len(TF_ids)])

        # df_in will be the expression matrix
        time_start = time.time()
        for row in df_in.iterrows():
            tmp_AUC_row = np.zeros(len(TF_ids))
            row_idx = row[0]
            # combined_gene_list = list(target_ids) + list(TF_ids)
            combined_gene_list = target_ids
            row_series_in = row[1][combined_gene_list]
            row_vec_in = row_series_in.values

            time_row_start = time.time()
            for tf_idx in range(len(TF_ids)):
                # weight_series_in = pd.Series(data = weight_mat[tf_idx, :], index = target_ids)
                # AUC_score = cal_AUC(row_series_in, weight_series_in)
                weight_vec_in = weight_mat[tf_idx, :]
                AUC_score = cal_AUC(row_vec_in, weight_vec_in)

                tmp_AUC_row[tf_idx] = AUC_score
            time_row_end = time.time()
            # print('label {}, row {} done in {:.2}s'.format(label, row_idx, time_row_end - time_row_start))
            AUC_mat[row_idx, :] = tmp_AUC_row
        time_end = time.time()
        print('label {} done in {:.2}s'.format(label, time_end - time_start))
            
        AUC_dict[label] = pd.DataFrame(data=AUC_mat, columns = TF_ids, index=original_index)
    
    return AUC_dict

# ----------
p2df = '/data/jianhao/hepatocyte_update_dataset_101619/magic_cell_mat_w_label_pd'
p2res = '/data/jianhao/hepatocyte_update_dataset_101619/new_results_with_monoc_NF_100'

normalized_weights, original_df, TF_ids, target_ids = normalized_by_target_norm(p2df, p2res)

AUC_dict = get_AUCell_mat(original_df, normalized_weights, TF_ids, target_ids)
with open('/data/jianhao/hepatocyte_update_dataset_101619/AUC_dict_hepa_monoc_states_all_target', 'wb') as f:
    pickle.dump(AUC_dict, f)
