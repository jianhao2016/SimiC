#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qizai <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
This is an implementation of simicLASSO
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import os
import argparse
import ipdb
import random
import copy
import itertools
import math
import sys
from sklearn.linear_model import LassoLars, MultiTaskLasso
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import NMF
from sklearn.metrics import accuracy_score, adjusted_mutual_info_score
from simiclasso.common_io import load_dataFrame, extract_df_columns, split_df_and_assignment
from simiclasso.evaluation_metric import get_r_squared
from simiclasso.sc_different_clustering import kmeans_clustering, nmf_clustering, spectral_clustering, evaluation_clustering
from simiclasso.gene_id_to_name import load_dict, save_dict
from scipy.linalg import eigh, eigvalsh

def extract_cluster_from_assignment(df_in, assignment, k_cluster):
    '''
    given DataFrame with all cells and all genes,
    provided a dictionary with {label: df_cluster}
    where df_cluster are selected by rows
    '''
    assign_np = np.array(assignment)
    cluster_dict = {}
    # for label in range(k_cluster):
    for label in set(assignment): 
        cluster_idx = np.where(assign_np == label)
        # df_in may not in order
        tmp = df_in.index.values
        df_cluster_index = tmp[cluster_idx]
        cluster_dict[label] = df_in.loc[df_cluster_index]
    return cluster_dict

def extract_tf_target_mat_from_cluster_dict(cluster_dict, tf_list, target_list):
    '''
    mat_dict is a dictionary:
        { label:
            { 'tf_mat': np.matrix,
              'target_mat': np.matrix}
        }
    tf_list is the input list,  but not every TFs may be included in DataFrame.
    TF_list is the matching list.
    '''
    mat_dict = {}
    for label in cluster_dict:
        cur_df = cluster_dict[label]
        TF_df = extract_df_columns(cur_df, tf_list)
        tmp_mat = TF_df.values
        # add fixed value 1 to last column, for the bias term in regression.
        m, n_x = tmp_mat.shape
        TF_mat = np.ones((m, n_x + 1))
        TF_mat[:, :-1] = tmp_mat
        TF_list = TF_df.columns.values.tolist()

        target_df = extract_df_columns(cur_df, target_list)
        target_mat = target_df.values
        target_list = target_df.columns.values.tolist()

        sys.stdout.flush()
        print('cell type ', label)
        print('\tTF size:', TF_mat.shape)
        print('\tTarget size:', target_mat.shape)
        mat_dict[label] = { 'tf_mat':TF_mat,
                            'target_mat':target_mat}
    return mat_dict, TF_list, target_list


def load_mat_dict_and_ids(df_in, tf_list, target_list, assignment, k_cluster):
    '''
    get clustered data matrix, as well as the TF_ids, and target genes ids
    '''
    cluster_dict = extract_cluster_from_assignment(df_in, assignment, k_cluster)
    mat_dict, TF_ids, target_ids = extract_tf_target_mat_from_cluster_dict(cluster_dict,
            tf_list, target_list)

    return mat_dict, TF_ids, target_ids

def loss_function_value(mat_dict, weight_dict, similarity, lambda1, lambda2):
    loss = 0
    num_labels = len(weight_dict.keys())
    for label in mat_dict.keys():
        Y_i = mat_dict[label]['target_mat']
        X_i = mat_dict[label]['tf_mat']
        W_i = weight_dict[label]
        m, n_x = X_i.shape
        loss += 1/m * (np.linalg.norm(Y_i - X_i @ W_i) ** 2)
        loss += lambda1 * np.linalg.norm(W_i, 1)
        # loss *= 1/m
        if similarity:
            # W_i = weight_dict[idx]
            # assert ((label + 1) % num_labels) in mat_dict.keys()
            if label != max(mat_dict.keys()):
                W_ip1 = weight_dict[label + 1]
            else:
                W_ip1 = W_i
            loss += lambda2 * (np.linalg.norm(W_i - W_ip1) ** 2)
    return loss

def std_error_per_cluster(mat_dict, weight_dict):
    std_error_dict = {}
    for label in mat_dict.keys():
        Y_i = mat_dict[label]['target_mat']
        X_i = mat_dict[label]['tf_mat']
        W_i = weight_dict[label]
        m, n_x = X_i.shape
        std_error = np.sqrt(1/m) * np.linalg.norm( Y_i - X_i @ W_i, axis = 0)
        std_error_dict[label] = std_error
    return std_error_dict



def get_gradient(mat_dict, weight_dict, label, similarity, lambda1, lambda2):
    '''
    get graident of loss function
    of W_k, weight matrix for cluster/label k
    '''
    Y_i = mat_dict[label]['target_mat']
    X_i = mat_dict[label]['tf_mat']
    W_i = weight_dict[label]
    m, n_x = X_i.shape

    last_label = max(weight_dict.keys())
    first_label = min(weight_dict.keys())

    grad_f = 2/m * X_i.T @ (X_i @ W_i - Y_i) + lambda1 * np.sign(W_i)
    # grad_f = one_step_gradient(X_i, W_i, Y_i, lambda1)
    if similarity:
        # [0, 1, 2], pick 0, then (0 -1) % 3 = 2, the last term
        # if pick 2, then (2+1) % 3 = 0, the first term
        num_labels = len(weight_dict.keys())
        if label == last_label:
            W_i_plus1 = W_i
        else:
            W_i_plus1 = weight_dict[(label + 1)]
        if label == first_label:
            W_i_minus1 = W_i
        else:
            W_i_minus1 = weight_dict[(label -1)]
        grad_f += 2 * lambda2 * (W_i - W_i_plus1 + W_i - W_i_minus1)
    return grad_f

def get_L_max(mat_dict, similarity, lambda1, lambda2):
    '''
    calculate the Lipschitz constant for each cluster/label
    '''
    L_max_dict = {}
    for label in mat_dict.keys():
        Y_i = mat_dict[label]['target_mat']
        X_i = mat_dict[label]['tf_mat']

        m, n_x = X_i.shape
        #### eigvals = (lo, hi) indexes of smallest and largest (in ascending order)
        #### eigenvalues and corresponding eigenvectors to be returned.
        #### 0 <= lo < hi <= M -1
        #### basically, eigh and eigvalsh doesn't make much of a difference
        L_tmp = 2/m * eigh(X_i.T @ X_i, eigvals = (n_x - 1, n_x - 1), eigvals_only = True)
        # L_tmp = 2/m * eigvalsh(X_i.T @ X_i, eigvals = (n_x - 1, n_x - 1), turbo = True)
        if similarity:
            L_tmp += 4 * lambda2
        L_max_dict[label] = L_tmp
    return L_max_dict

def average_r2_score(mat_dict, weight_dict):
    '''
    calculate the total avereage R2 squared score for all clusters
    i.e.
        average_r2 = sum(r2 for each cluster) / num_cluster
    '''
    num_cluster = len(weight_dict.keys())
    sum_r2 = 0
    r2_dict = {}
    for label in weight_dict:
        X_i = mat_dict[label]['tf_mat']
        Y_i = mat_dict[label]['target_mat']
        W_i = weight_dict[label]
        m, n_x = X_i.shape

        Y_pred = X_i @ W_i

        # # ordinary R2
        # sum_r2 += get_r_squared(Y_i, Y_pred, k)

        # adjusted R2
        # num_idpt_variable = min(n_x, m - 2)
        W_i_avg = np.mean(W_i, axis = 1)
        W_avg_count = np.count_nonzero(W_i_avg > 1e-3)
        num_idpt_variable = min(m, W_avg_count) - 2
        list_of_r_squared = get_r_squared(Y_i, Y_pred, k = num_idpt_variable,
                multioutput = 'raw_values')
        r2_dict[label] = list_of_r_squared
        sum_r2 += np.mean(list_of_r_squared)

    aver_r2 = sum_r2 / num_cluster
    return aver_r2, r2_dict


def rcd_lasso_multi_cluster(mat_dict, similarity,
        lambda1 = 1e-3, lambda2 = 1e-3,
        slience = False, max_rcd_iter = 50000):
    L_max_dict = get_L_max(mat_dict, similarity, lambda1, lambda2)

    weight_dict = {}
    # initialize weight dict for each label/cluster
    for label in mat_dict.keys():
        _, n_x = mat_dict[label]['tf_mat'].shape
        m, n_y = mat_dict[label]['target_mat'].shape
        # tmp = np.random.randn(n_x, n_y)
        tmp = np.zeros((n_x, n_y))
        weight_dict[label] = tmp

    weight_dict_0 = copy.deepcopy(weight_dict)
    loss_0 = loss_function_value(mat_dict, weight_dict, similarity,
            lambda1, lambda2)
    r2_0, r2_dict_0 = average_r2_score(mat_dict, weight_dict)

    if not slience:
        sys.stdout.flush()
        print('strat RCD process...')
        print('-' * 7)
        print('\tloss w. reg before RCD: {:.4f}'.format(loss_0))
        print('\tR squared before RCD: {:.4f}'.format(r2_0))
        print('-' * 7)

    loss_old = loss_0
    num_iter = 0
    label_list = list(mat_dict)
    time_sum = 0
    pause_step = 50000
    while num_iter <= max_rcd_iter:
        num_iter += 1
        if num_iter % pause_step == 0 and not slience:
            t1 = time.time()
            loss_tmp = loss_function_value(mat_dict, weight_dict, similarity,
                    lambda1, lambda2)
            t2 = time.time()
            sys.stdout.flush()
            print('\ttime elapse in eval: {:.4f}s'.format(t2 - t1))
            print('\ttime elapse in update: {:.4f}s'.format(time_sum / pause_step))
            time_sum = 0

            print('\titeration {}, loss w. reg = {:.4f}'.format(num_iter, loss_tmp))
            print('-' * 7)

        t1 = time.time()
        label = random.choice(label_list)
        step_size = 1 / L_max_dict[label]
        grad_f = get_gradient(mat_dict, weight_dict, label, similarity,
                lambda1, lambda2)
        weight_dict[label] -= step_size * grad_f
        t2 = time.time()
        time_sum += (t2 - t1)

    loss_final = loss_function_value(mat_dict, weight_dict, similarity,
            lambda1, lambda2)
    r2_final, r2_dict_final = average_r2_score(mat_dict, weight_dict)
    if not slience:
        sys.stdout.flush()
        print('\tloss w. reg after RCD: {:.4f}'.format(loss_final))
        print('\tR squared after RCD: {:.4f}'.format(r2_final))
        print('-' * 7)
        print('Done RCD process!')

    return weight_dict, weight_dict_0

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
        sys.stdout.flush()
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
    sum_list = df_target.var().values
    idx = np.argpartition(sum_list, -top_k)
    # print(idx)
    important_target_list = []
    for i in idx[-top_k:]:
        important_target_list.append(new_target_list[int(i)])
    # print(df[important_target_list].sum())
    return important_target_list

def get_top_k_MAD_TFs(top_k, df, tf_list):
    '''
    tf_list: list of all TF, in 'TF_symbol_pickle'
    '''
    df_tf = extract_df_columns(df, tf_list)
    new_tf_list = df_tf.columns.values.tolist()
    MAD_df = df_tf.mad(axis = 0)
    top_k_tfs = MAD_df.nlargest(top_k).index.values.tolist()
    return top_k_tfs
    


def cross_validation(mat_dict_train, similarity, list_of_l1, list_of_l2):
    '''
    perform cross validation on training set
    '''
    opt_r2_score = float('-inf')
    k = 5
    for lambda1, lambda2 in itertools.product(list_of_l1,
            list_of_l2):
        # run k-fold evaluation
        r2_tmp = k_fold_evaluation(k, mat_dict_train,
                similarity, lambda1, lambda2)
        sys.stdout.flush()
        print('lambda1 = {}, lamda2 = {}, done'.format(lambda1, lambda2))
        print('----> adjusetd R2 = {:.4f}'.format(r2_tmp))
        print('////////')
        if r2_tmp > opt_r2_score:
            l1_best, l2_best = lambda1, lambda2
            opt_r2_score = r2_tmp

    return l1_best, l2_best, opt_r2_score

def k_fold_evaluation(k, mat_dict_train, similarity, lambda1, lambda2):
    '''
    split training data set into equal k fold.
    train on (k - 1) and test on rest.
    r2_tmp = average r2 from each evalutaion
    '''
    r2_tmp = 0
    for idx in range(k):
        mat_dict_train, mat_dict_eval = get_train_mat_in_k_fold(mat_dict_train,
                idx, k)
        weight_dict_trained, _ = rcd_lasso_multi_cluster(mat_dict_train, similarity,
                lambda1, lambda2, slience = True, max_rcd_iter = 10000)
        r2_aver, _ = average_r2_score(mat_dict_eval, weight_dict_trained)
        r2_tmp += r2_aver

    aver_r2_tmp = r2_tmp/k
    return aver_r2_tmp

def get_train_mat_in_k_fold(mat_dict, idx, k):
    '''
    split input: mat_dict
    into k folder, select idx as test, rest as train
    '''
    mat_dict_train, mat_dict_eval = {}, {}
    for label in mat_dict:
        X_i = mat_dict[label]['tf_mat']
        Y_i = mat_dict[label]['target_mat']
        _, n_x = X_i.shape

        concat_XY = np.hstack((X_i, Y_i))
        split_XY = np.array_split(concat_XY, k)
        eval_x, eval_y = split_XY[idx][:, :n_x], split_XY[idx][:, n_x:]
        mat_dict_eval[label] = {
                'tf_mat': eval_x,
                'target_mat': eval_y
                }

        del split_XY[idx]
        tmp_XY = np.vstack(split_XY)
        train_x, train_y = tmp_XY[:, :n_x], tmp_XY[:, n_x:]
        mat_dict_train[label] = {
                'tf_mat': train_x,
                'target_mat': train_y
                }
    return mat_dict_train, mat_dict_eval

def simicLASSO_op(p2df, p2assignment, similarity, p2tf, p2saved_file, 
        k_cluster = None, num_TFs = -1, num_target_genes = -1, 
        numIter = 1000, _NF = 1, lambda1 = 1e-2, lambda2 = 1e-5,
        cross_val = False, num_rep = 1, max_rcd_iter = 500000, 
        df_with_label = True):
    '''
    perform the GRN inference algorithm, simicLASSO.
    args:
        p2df: path to dataframe
            dataframe should be:
                    gene1, gene2, ..., genek, label
            cell1:  x,     x,   , ..., x,   , type1
            cell2:  x,     x,   , ..., x,   , type2

        p2assignment: path to clustering order assignment file.
                      a text file with each line corresponding to cell order.
        k_cluster: number of cluster in dataset.
        similarity: 1 - Yes, 0 - No.
        p2tf: path to list of all TFs.
        num_TFs: number of TFs used in regression.
        num_target_genes: number of target genes used in regression
        num_rep: number of repeated test.
    output: 
        save the weight dictionary and gene list to p2saved_file.
    '''

    if p2df == None:
        raise ValueError('please enter the path to dataframe file saved from load_data.py')
    elif os.path.isfile(p2df):
        p2df_file = p2df
    else:
        raise ValueError('{} is not a valid file'.format(p2df))

    original_df = pd.read_pickle(p2df)
    if df_with_label:
        original_feat_cols = list(original_df.columns[:-1])
    else:
        original_feat_cols = list(original_df.columns)

    X_raw, Y, feat_cols = load_dataFrame(p2df_file, original_feat_cols, df_with_label)
    X = normalize(X_raw) * _NF
    len_of_gene = len(feat_cols)
    # X = X_raw

    n_dim, m_dim = X.shape

    clustering_method = kmeans_clustering
    # get centroids and assignment from clustering
    #
    if p2assignment != None:
        p2assign_file = p2assignment
        if os.path.isfile(p2assign_file):
            assignment = []
            with open(p2assign_file, 'r') as f:
                for line in f:
                    label = line.split()[0]
                    assignment.append(int(label))
            assignment = np.array(assignment)
    else:
        sys.stdout.flush()
        print('invalid assignment file, use clustering assignment.')
        if k_cluster is None:
            raise ValueError('assignment file is not provided, and number of cluster is not given. quit funciton now.')
        D_final, assignment = clustering_method(X, k_cluster, numIter)
        if df_with_label:
            acc, AMI = evaluation_clustering(assignment, Y)
            sys.stdout.flush()
            print('clustering accuracy = {:.4f}'.format(acc))
            print('AMI = {:.4f}'.format(AMI))
            print('D_final = ', D_final)
            print('shape of D mat:', D_final.shape)
        else:
            print('no label provided in dataframe, accuracy not available')


    #### BEGIN of the regression part
    sys.stdout.flush()
    print('------ Begin the regression part...')
    df = pd.read_pickle(p2df_file)
    # the following should be moved to load_dataset.py
    # if expr_name == 'mouse' or expr_name == 'human_mgh':
    #     df = df.reset_index(drop=True)

    # if expr_name == 'human_mgh':
    #     df = df.rename(columns = dict((k, k.replace("'", '')) for k in df.columns))
    print('df in regression shape = ', df.shape)


    df = df.reset_index(drop=True)
    # df[feat_cols] = X
    df_train, df_test, assign_train, assign_test = split_df_and_assignment(df, assignment)
    sys.stdout.flush()
    print('df test = ', df_test.shape)
    print('test data assignment set:', set(list(assign_test)))
    print('df train = ', df_train.shape)
    print('train data assignment set:', set(list(assign_train)))
    print('-' * 7)

    # p2tf_gene_list = os.path.join(gene_list_root, 'gene_id_TF_and_ensembl_pickle')
    if os.path.isfile(p2tf):
        p2tf_gene_list = p2tf
    else:
        raise ValueError('invalid path to tf list file.')

    # p2tf_gene_list = os.path.join(gene_list_root, 'top_200_MAD_val_selected_TF_pickle')
    # gene_list_root = '../../data/diff_gene_list'
    # p2target_gene_list = os.path.join(gene_list_root, 'gene_id_non_TF_and_ensembl_pickle')


    with open(p2tf_gene_list, 'rb') as f:
        full_tf_list = pickle.load(f)
        full_tf_list = list(full_tf_list)
    # with open(p2target_gene_list, 'rb') as f:
    #     target_list = pickle.load(f)
    #     target_list = list(target_list)

    # if gene_list_type == 'symbol':
    #     with open('../../data/merged_gene_id_to_name_pickle', 'rb') as f:
    #         ENSG_2_symbol_dict = pickle.load(f)
    #     # symbol_2_ENSG_dict = dict((v, k) for k, v in ENSG_2_symbol_dict.items())
    #     # tf_list = [ENSG_2_symbol_dict[a] for a in tf_list if a in ENSG_2_symbol_dict]
    #     target_list = [ENSG_2_symbol_dict[a] for a in target_list if a in ENSG_2_symbol_dict]

    full_tf_list_lower_case = [x.lower() for x in full_tf_list]
    target_list = [x for x in feat_cols if x.lower() not in full_tf_list_lower_case]

    num_TFs = min(num_TFs, len(full_tf_list))
    num_target_genes = min(num_target_genes, len(target_list))

    if num_TFs != -1:
        tf_list = get_top_k_MAD_TFs(num_TFs, df_train, full_tf_list)
    else:
        tf_list = get_top_k_MAD_TFs(len(full_tf_list), df_train, full_tf_list)

    if num_target_genes != -1:
        query_target_list = get_top_k_non_zero_targets(num_target_genes, df_train, target_list)
    else:
        query_target_list = get_top_k_non_zero_targets(len(target_list), df_train, target_list)

    sys.stdout.flush()
    print('-' * 7)

    print('.... generating train set')
    mat_dict_train, TF_ids, target_ids = load_mat_dict_and_ids(df_train, tf_list,
            query_target_list, assign_train, k_cluster)
    print('-' * 7)

    print('.... generating test set')
    mat_dict_test, _, _ = load_mat_dict_and_ids(df_test, tf_list, query_target_list,
            assign_test, k_cluster)
    print('-' * 7)

    if cross_val == True:
        ### run cross_validation!!!!!! #############

        sys.stdout.flush()
        print('start cross validation!!!')
        list_of_l1 = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        # list_of_l1 = [1e-1, 1e-2, 1e-3]
        list_of_l2 = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
        l1_opt, l2_opt, r2_opt = cross_validation(mat_dict_train, similarity, list_of_l1, list_of_l2)
        print('cv done! lambda1 = {}, lambda2 = {}, opt R squared on eval {:.4f}'.format(
            l1_opt, l2_opt, r2_opt))
        print('-' * 7)
        lambda1 = l1_opt
        lambda2 = l2_opt
        ############### end of cv ####################


    #### optimize using RCD
    # ipdb.set_trace()
    train_error, test_error = [], []
    r2_final_train, r2_final_test = [], []
    r2_final_0 = 0
    for _ in range(num_rep):
        trained_weight_dict, weight_dict_0 = rcd_lasso_multi_cluster(mat_dict_train, similarity,
                                        lambda1, lambda2, slience = True, max_rcd_iter = max_rcd_iter)


        test_error.append(loss_function_value(mat_dict_test, trained_weight_dict, similarity,
                lambda1 = 0, lambda2 = 0))
        train_error.append(loss_function_value(mat_dict_train, trained_weight_dict, similarity,
                lambda1 = 0, lambda2 = 0))
        std_error_dict_test = std_error_per_cluster(mat_dict_test, trained_weight_dict)
        # ipdb.set_trace()
        r2_final_train.append(average_r2_score(mat_dict_train, trained_weight_dict)[0])
        r2_aver_test, r2_dict_test = average_r2_score(mat_dict_test, trained_weight_dict)
        r2_final_test.append(r2_aver_test)
        r2_final_0 += average_r2_score(mat_dict_test, weight_dict_0)[0]
    sys.stdout.flush()
    print('-' * 7)
    # print(train_error)
    print('final train error w.o. reg = {:.4f}+/-{:.4f}'.format(np.mean(train_error),
                                                                np.std(train_error)))
    print('test error w.o. reg = {:.4f}+/-{:.4f}'.format(np.mean(test_error),
                                                         np.std(test_error)))

    print('-' * 7)
    print('R squared of test set(before): {:.4f}'.format(r2_final_0/num_rep))

    print('R squared of train set(after): {:.4f}+/-{:.4f}'.format(np.mean(r2_final_train),
                                                                  np.std(r2_final_train)))
    print('R squared of test set(after): {:.4f}+/-{:.4f}'.format(np.mean(r2_final_test),
                                                                 np.std(r2_final_test)))

    # # ipdb.set_trace()
    # # variable_list[1] *= 1.1
    # root_path = '../../data/'
    # path_to_gene_name_dict = os.path.join(root_path, 'merged_gene_id_to_name_pickle')
    # gene_id_name_mapping = load_dict(path_to_gene_name_dict)

    dict_to_saved = {'weight_dic' : trained_weight_dict,
                     'adjusted_r_squared': r2_dict_test,
                     'standard_error': std_error_dict_test,
                     # 'TF_ids'     : [symbols.upper() for symbols in TF_ids],
                     # 'query_targets' : [symbols.upper() for symbols in query_target_list]
                     'TF_ids'     : [symbols for symbols in TF_ids],
                     'query_targets' : [symbols for symbols in query_target_list]
                     }

    save_dict(dict_to_saved, p2saved_file)

