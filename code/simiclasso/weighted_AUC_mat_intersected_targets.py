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
import scipy.stats as st
import ipdb
import time
import sys
import os

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
    # original_df.columns = map(str.upper, original_df.columns)

    target_df = original_df[target_ids]
    target_norms = np.linalg.norm(target_df, axis = 0)

    normalized_weights = {}
    for label in weight_dic:
        normalized_mat = weight_dic[label] / target_norms
        normalized_weights[label] = normalized_mat[:len(TF_ids), :]

    return normalized_weights, original_df, TF_ids, target_ids

def find_significant_weights(normalized_weight_matrix, th = 0.9):
    z_scores = st.zscore(normalized_weight_matrix, axis = 1)
    cdfs = st.norm.cdf(z_scores)
    outside_90_intervals = np.logical_or(cdfs > th, cdfs < 1 - th)
    inside_90_intervals = np.logical_and(cdfs <= th, cdfs >= 1 - th)
    significant_weights_indicator = outside_90_intervals
    return significant_weights_indicator

def find_intersect_and_union_targets(normalized_by_target_norm, TF_ids, target_ids):
    intersect_indicator = np.ones((len(TF_ids), len(target_ids)))
    union_indicator = np.zeros((len(TF_ids), len(target_ids)))
    
    sig_ind_dict = {}
    sig_target_ids_dict = {}
    
    for states in normalized_by_target_norm:
        normalized_mat = normalized_by_target_norm[states]
        sig_ind = find_significant_weights(normalized_mat)
        intersect_indicator = np.logical_and(intersect_indicator, sig_ind)
        union_indicator = np.logical_or(union_indicator, sig_ind)
        
        # sig_ind_dict and sig_target_ids_dict contains the significant weight indicator for each states.
        # each of the size #TF \times #target.
        # sig_target_ids_dict is a dictionary, which has each state as key.
        # for each key, the value is a dictionary of length #TF, it contains the index of targets. 
        # NOT True/False indicator, 
        # as well as the actual Target Names.
        sig_ind_dict[states] = sig_ind
        sig_target_ids_dict[states] = {}
        for idx, tf in enumerate(TF_ids):
            weight_ind_per_tf = sig_ind[idx]
            sig_target_idx = np.where(weight_ind_per_tf == 1)[0]
            sig_target_ids_dict[states][tf] = {'target_index': sig_target_idx,
                                                'target_names': [target_ids[x] for x in sig_target_idx]}
    
    # when we have went through all states, the intersection indicator shows the target locations 
    # across all states. 
    interected_targets_dict = {}
    for idx, tf in enumerate(TF_ids):
        intersect_row = intersect_indicator[idx]
        intersect_target_idx = np.where(intersect_row == 1)[0]
        intersect_target_names = [target_ids[x] for x in intersect_target_idx]
        interected_targets_dict[tf] = {'target_names': intersect_target_names,
                                      'target_index': intersect_target_idx}
        
        
    return sig_ind_dict, sig_target_ids_dict, interected_targets_dict, intersect_indicator

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

def cal_AUC(row_vec_in, weight_vec_in, percent_of_target = 1):
    row_np = np.array(row_vec_in)
    weight_np = np.array(weight_vec_in)
    expression_descending_order = np.argsort(row_np)[::-1]
    ordered_weighted = weight_np[expression_descending_order]

    # div_factor = 1
    # len_of_genes = len(ordered_weighted) // div_factor
    len_of_genes = int(len(ordered_weighted) * percent_of_target)
    sum_of_weight = np.sum(ordered_weighted[:len_of_genes])

    running_sum = [np.sum(ordered_weighted[:x+1]) for x in range(len_of_genes)]
    if len_of_genes > 0 and sum_of_weight > 0:
        AUC_score = np.sum(running_sum) / (sum_of_weight * len_of_genes)
    else:
        AUC_score = 0
    return AUC_score


def get_AUCell_mat(original_df, weight_dict, TF_ids, target_ids, intersected_targets_dict, percent_of_target = 1):
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
        row_print_count = 0
        time_row_start = time.time()
        for row in df_in.iterrows():
            row_print_count += 1
            tmp_AUC_row = np.zeros(len(TF_ids))
            row_idx = row[0]
            # combined_gene_list = list(target_ids) + list(TF_ids)
            combined_gene_list = target_ids
            row_series_in = row[1][combined_gene_list]
            row_vec_in = row_series_in.values

            for tf_idx in range(len(TF_ids)):
                # weight_series_in = pd.Series(data = weight_mat[tf_idx, :], index = target_ids)
                # AUC_score = cal_AUC(row_series_in, weight_series_in)
                weight_vec_in = weight_mat[tf_idx, :]

                intersected_target_index = intersected_targets_dict[TF_ids[tf_idx]]['target_index']

                selected_row_vec_in = np.zeros_like(row_vec_in)
                # selected_row_vec_in[intersected_target_index] = row_vec_in[intersected_target_index]
                selected_row_vec_in = row_vec_in

                selected_weight_vec_in = np.zeros_like(weight_vec_in)
                selected_weight_vec_in[intersected_target_index] = weight_vec_in[intersected_target_index]

                AUC_score = cal_AUC(selected_row_vec_in, selected_weight_vec_in, percent_of_target)

                tmp_AUC_row[tf_idx] = AUC_score
            if row_print_count % 1000 == 0:
                time_row_end = time.time()
                print('label {}, 1000 row done in {:.2}s'.format(label, time_row_end - time_row_start))
                time_row_start == time.time()
            AUC_mat[row_idx, :] = tmp_AUC_row
        time_end = time.time()
        sys.stdout.flush()
        print('label {} done in {:.2}s'.format(label, time_end - time_start))
            
        AUC_dict[label] = pd.DataFrame(data=AUC_mat, columns = TF_ids, index=original_index)
    
    return AUC_dict

def find_interseciton_of_genes_in_all_states_for_each_TF(normalized_weight):
    '''
    in each state, for every TF, find the set of target genes falls in z-test.
    Then find the intersection of these set of genes across states.
    output:
        >> for each TF there is a list of target genes. Instead of all genes.
    '''
    pass

def main_fn(p2df, p2res, p2saved_file, percent_of_target = 1):
    normalized_weights, original_df, TF_ids, target_ids = normalized_by_target_norm(p2df, p2res)

    _, _, intersected_targets_dict, _ = find_intersect_and_union_targets(normalized_weights, TF_ids, target_ids)
    
    AUC_dict = get_AUCell_mat(original_df, normalized_weights, TF_ids, target_ids, intersected_targets_dict, percent_of_target)
    with open(p2saved_file, 'wb') as f:
        pickle.dump(AUC_dict, f)

if __name__ == '__main__':
    # ----------
    # p2df = '/data/jianhao/hepatocyte_update_dataset_101619/magic_cell_mat_w_label_pd'
    # p2res = '/data/jianhao/hepatocyte_update_dataset_101619/new_results_with_monoc_NF_100'
    # p2df = '/data/jianhao/hepatocyte_update_dataset_101619/Bee_DF.pickle'
    # p2res = '/data/jianhao/hepatocyte_update_dataset_101619/Bees_results_updated'
    # p2saved_file = '/data/jianhao/hepatocyte_update_dataset_101619/AUC_dict_bees'

    cima_res = '/data/cima/hsc_simic_1.6_100_2000_MAD.pickle'
    cima_df = '/data/cima/hsc_DF_for_simic_100_2000_MAD.pickle'
    cima_saved_AUC = '/data/cima/intersected_AUC_cima'
    main_fn(cima_df, cima_res, cima_saved_AUC)
    print('-' * 7, 'cima done!')

    res_list = ['monoc_simple_state_results_NF_100', 'hepa_results_monoc_simple_states_no_simic', 
            'Bees_results_simic', 'Bees_results_no_simic']

    df_list = ['magic_cell_mat_w_label_pd', 'magic_cell_mat_w_label_pd',
            'Bee_DF.pickle', 'Bee_DF.pickle']

    file_list = ['intersected_AUC_dict_monoc_3_states_hepa_simic',
            'intersected_AUC_dict_monoc_3_states_hepa_no_simic', 
            'intersected_AUC_dict_Bee_simic', 
            'intersected_AUC_dict_Bee_no_simic']

    for idx in range(4):
        root_path = '/data/jianhao/hepatocyte_update_dataset_101619/'
        p2res = os.path.join(root_path, res_list[idx])
        p2df = os.path.join(root_path, df_list[idx])
        p2saved_file = os.path.join(root_path, 'intersect_same_length_weight', file_list[idx])

        print(p2saved_file)
        main_fn(p2df, p2res, p2saved_file)
    
