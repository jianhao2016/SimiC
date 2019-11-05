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
from common_io import load_dataFrame, extract_df_columns, split_df_and_assignment
from evaluation_metric import get_r_squared
from sc_different_clustering import kmeans_clustering, nmf_clustering, spectral_clustering, evaluation_clustering
from gene_id_to_name import load_dict, save_dict
from scipy.linalg import eigh, eigvalsh

def extract_cluster_from_assignment(df_in, assignment, k_cluster):
    '''
    given DataFrame with all cells and all genes,
    provided a dictionary with {label: df_cluster}
    where df_cluster are selected by rows
    '''
    assign_np = np.array(assignment)
    cluster_dict = {}
    for label in range(k_cluster):
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
        TF_mat = TF_df.values
        TF_list = TF_df.columns.values.tolist()

        target_df = extract_df_columns(cur_df, target_list)
        target_mat = target_df.values
        target_list = target_df.columns.values.tolist()

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
            assert ((label + 1) % num_labels) in mat_dict.keys()
            W_ip1 = weight_dict[(label + 1) % num_labels]
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

    grad_f = 2/m * X_i.T @ (X_i @ W_i - Y_i) + lambda1 * np.sign(W_i)
    # grad_f = one_step_gradient(X_i, W_i, Y_i, lambda1)
    if similarity:
        # [0, 1, 2], pick 0, then (0 -1) % 3 = 2, the last term
        # if pick 2, then (2+1) % 3 = 0, the first term
        num_labels = len(weight_dict.keys())
        W_i_minus1 = weight_dict[(label -1) % num_labels]
        W_i_plus1 = weight_dict[(label + 1) % num_labels]
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
        slience = False):
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
    while num_iter <= 500000:
        num_iter += 1
        if num_iter % pause_step == 0 and not slience:
            t1 = time.time()
            loss_tmp = loss_function_value(mat_dict, weight_dict, similarity,
                    lambda1, lambda2)
            t2 = time.time()
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
                lambda1, lambda2, slience = True)
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

def simicLASSO_op(p2df, p2fc, p2assignment, k_cluster, similarity, p2tf, 
        p2saved_file, num_target_genes, gene_list_type = 'symbol', 
        numIter = 1000, _NF = 1, lambda1 = 1e-2, lambda2 = 1e-5,
        cross_val = False, num_rep = 1):
    '''
    perform the GRN inference algorithm, simicLASSO.
    args:
        p2df: path to dataframe
        p2fc: path to feature column
        p2assignment: path to clustering order assignment file.
                      a text file with each line corresponding to cell order.
        k_cluster: number of cluster in dataset.
        similarity: 1 - Yes, 0 - No.
        p2tf: path to list of TF used in regression.
        num_target_genes: number of target genes used in regression
        gene_list_type: annotation used in feature column, either 'symbol',
                        as 'GATA6', or 'ensembl', as 'ENSGxxxxxx'
        num_rep: number of repeated test.
    output: 
        save the weight dictionary and gene list to p2saved_file.
    '''
    if gene_list_type not in ['symbol', 'ensembl']:
        raise ValueError('gene_list_type must be either "symbol" or "ensembl"')

    if p2df == None:
        raise ValueError('please enter the path to dataframe file saved from load_data.py')
    elif os.path.isfile(p2df):
        p2df_file = p2df
    else:
        raise ValueError('{} is not a valid file'.format(p2df))

    if p2fc == None:
        raise ValueError('please enter the path to feature column file')
    elif os.path.isfile(p2fc):
        p2feat_file = p2fc
    else:
        raise ValueError('{} is not a valid file'.format(p2fc))

    with open(p2feat_file, 'rb') as f:
        original_feat_cols = pickle.load(f)

    X_raw, Y, feat_cols = load_dataFrame(p2df_file, original_feat_cols)
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
            print('invalid assignment file, use clustering assignment.')
            D_final, assignment = clustering_method(X, k_cluster, numIter)
            acc, AMI = evaluation_clustering(assignment, Y)
            print('clustering accuracy = {:.4f}'.format(acc))
            print('AMI = {:.4f}'.format(AMI))
            print('D_final = ', D_final)
            print('shape of D mat:', D_final.shape)


    #### BEGIN of the regression part
    print('------ Begin the regression part...')
    df = pd.read_pickle(p2df_file)
    # the following should be moved to load_dataset.py
    # if expr_name == 'mouse' or expr_name == 'human_mgh':
    #     df = df.reset_index(drop=True)

    # if expr_name == 'human_mgh':
    #     df = df.rename(columns = dict((k, k.replace("'", '')) for k in df.columns))
    print('df in regression shape = ', df.shape)


    df = df.reset_index(drop=True)
    df[feat_cols] = X
    df_train, df_test, assign_train, assign_test = split_df_and_assignment(df, assignment)
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
    gene_list_root = '../../data/diff_gene_list'
    p2target_gene_list = os.path.join(gene_list_root, 'gene_id_non_TF_and_ensembl_pickle')


    with open(p2tf_gene_list, 'rb') as f:
        tf_list = pickle.load(f)
        tf_list = list(tf_list)
    with open(p2target_gene_list, 'rb') as f:
        target_list = pickle.load(f)
        target_list = list(target_list)

    if gene_list_type == 'symbol':
        with open('../../data/merged_gene_id_to_name_pickle', 'rb') as f:
            ENSG_2_symbol_dict = pickle.load(f)
        # symbol_2_ENSG_dict = dict((v, k) for k, v in ENSG_2_symbol_dict.items())
        # tf_list = [ENSG_2_symbol_dict[a] for a in tf_list if a in ENSG_2_symbol_dict]
        target_list = [ENSG_2_symbol_dict[a] for a in target_list if a in ENSG_2_symbol_dict]

    query_target_list = get_top_k_non_zero_targets(num_target_genes, df_train, target_list)
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
                                        lambda1, lambda2, slience = True)


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
    root_path = '../../data/'
    path_to_gene_name_dict = os.path.join(root_path, 'merged_gene_id_to_name_pickle')
    gene_id_name_mapping = load_dict(path_to_gene_name_dict)

    dict_to_saved = {'weight_dic' : trained_weight_dict,
                     'adjusted_r_squared': r2_dict_test,
                     'standard_error': std_error_dict_test,
                     'TF_ids'     : [symbols.upper() for symbols in TF_ids],
                     'query_targets' : [symbols.upper() for symbols in query_target_list]
                     }

    save_dict(dict_to_saved, p2saved_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--numIter', type=int, default=1000)
    parser.add_argument('--method', type=str, choices=['kmeans', 'nmf', 'spectral'],
            default = 'kmeans')
    parser.add_argument('--NF', type=float, default=1, help='normalization factor')

    parser.add_argument('--num_cells', type=int, default=500,
            help='number of cells per cluster')
    parser.add_argument('--num_target_genes', type=int, default=10,
            help='number of target genes')

    parser.add_argument('--p2df', type=str, default = None,
            help = 'path to dataframe')
    parser.add_argument('--p2fc', type=str, default = None,
            help = 'path to feature column')
    parser.add_argument('--p2assignment', type = str, default = None,
            help = 'path to ground truth assignment file')
    parser.add_argument('--p2tf', type = str, default = None,
            help = 'path to list of TFs')
    parser.add_argument('--k_cluster', type=int, default = None,
            help='number of clusters in the data, default None')
    parser.add_argument('--gene_list_type', type = str,
            choices = ['ensembl', 'symbol'], default = 'ensembl',
            help = 'gene annotation used in gene list')

    parser.add_argument('--similarity', type=int, choices=[0, 1], default=1,
            help='0: no similarity constraint, 1: add similarity')
    parser.add_argument('--lambda1', type=float, default=1e-2,
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
    cells_per_cluster = args.num_cells
    similarity = (args.similarity == 1)
    expr_dtype = args.data_type
    num_target = args.num_target_genes
    # cima = False
    expr_name = 'human_mgh'
    # set_of_gene = 'TFs'
    set_of_gene = 'human_mgh'


    if expr_name == 'cima':
        data_root = '/data/cima/'
        k_cluster = 13
        p2feat_file = os.path.join(data_root, 'df_feature_column_cima')
        if expr_dtype == 'raw':
            p2df_file = os.path.join(data_root, 'pandas_dataframe_cima')
        elif expr_dtype == 'magic':
            p2df_file = os.path.join(data_root, 'pandas_dataframe_cima_magic')
        else:
            print('cima does not support data type:', expr_dtype)
            raise ValueError
    elif expr_name == 'mouse':
        data_root = '/data/jianhao/clus_GRN'
        k_cluster = 7
        if expr_dtype == 'raw':
            raise ValueError('mouse dont have raw')
        elif expr_dtype == 'magic':
            file_name_df = 'pandas_dataframe_mouse_magic'
            p2df_file = os.path.join(data_root, 'magic_expression_data', file_name_df)
    elif expr_name == 'human_mgh':
        data_root = '/data/jianhao/clus_GRN'
        k_cluster = 6
        if expr_dtype == 'raw':
            file_name_df = 'pandas_dataframe_human_MGH'
            p2df_file = os.path.join(data_root, 'raw_expression_data', file_name_df)
        elif expr_dtype == 'magic':
            file_name_df = 'pandas_dataframe_human_MGH_magic'
            p2df_file = os.path.join(data_root, 'magic_expression_data', file_name_df)
    else:
        data_root = '/data/jianhao/clus_GRN'
        k_cluster = 3

        if expr_dtype == 'raw':
            p2df_file = os.path.join(data_root, 'raw_expression_data', 'pandas_dataframe_{}'.format(cells_per_cluster))
        elif expr_dtype == 'magic':
            file_name_df = 'magic_smoothed_all_genes_{}_cells_per_cluster'.format(cells_per_cluster)
            p2df_file = os.path.join(data_root, 'magic_expression_data', file_name_df)
        elif expr_dtype == 'rwr':
            file_name_df = 'smoothed_dataframe_{}_cells_per_cluster_all_gene_coexpression'.format(cells_per_cluster)
            p2df_file = os.path.join(data_root, 'ensembl94', file_name_df)

    if args.k_cluster != None:
        k_cluster = args.k_cluster

    if set_of_gene == 'all':
        file_name_feat_cols = 'df_feature_column'
    elif set_of_gene == 'landmark':
        file_name_feat_cols = 'df_feature_column_lm'
    elif set_of_gene == 'ensembl':
        file_name_feat_cols = 'ensembl_gene_list'
    elif set_of_gene == 'encode_v29':
        file_name_feat_cols = 'gene_id_list_encode_v29_release_pickle'
    elif set_of_gene == 'not_encode_v29':
        file_name_feat_cols = 'gene_id_list_not_in_encode_v29_release_pickle'
    elif set_of_gene == 'TFs':
        file_name_feat_cols = 'TF_ids_human_protein_atlas_pickle'
    # next two condition consider only ensembl genes
    elif set_of_gene == 'TF_and_ensembl':
        file_name_feat_cols = 'gene_id_TF_and_ensembl_pickle'
    elif set_of_gene == 'not_TF_and_ensembl':
        file_name_feat_cols = 'gene_id_non_TF_and_ensembl_pickle'
    elif set_of_gene == 'mouse':
        file_name_feat_cols = 'df_feature_column_mouse'
    elif set_of_gene == 'human_mgh':
        file_name_feat_cols = 'df_feature_column_human_MGH'

    gene_list_root = '/data/jianhao/clus_GRN/diff_gene_list'
    p2feat_file = os.path.join(gene_list_root, file_name_feat_cols)

    if args.p2df == None:
        raise ValueError('please enter the path to dataframe file saved from load_data.py')
    elif os.path.isfile(args.p2df):
        p2df_file = args.p2df
    else:
        raise ValueError('{} is not a valid file'.format(args.p2df))

    if args.p2fc == None:
        raise ValueError('please enter the path to feature column file')
    elif os.path.isfile(args.p2fc):
        p2feat_file = args.p2fc
    else:
        raise ValueError('{} is not a valid file'.format(args.p2fc))







    with open(p2feat_file, 'rb') as f:
        original_feat_cols = pickle.load(f)

    # tmp_df = pd.read_pickle(os.path.join(data_root, 'ensembl94', 'smoothed_dataframe_100_cells_per_cluster_all_gene_coexpression'))
    # ensembl_list = list(tmp_df.columns.values)

    # intersection = set.intersection(set(original_feat_cols), set(ensembl_list))
    # original_feat_cols = list(intersection)

    # ipdb.set_trace()
    X_raw, Y, feat_cols = load_dataFrame(p2df_file, original_feat_cols)
    X = normalize(X_raw) * _NF
    len_of_gene = len(feat_cols)
    # X = X_raw

    n_dim, m_dim = X.shape

    if method == 'kmeans':
        clustering_method = kmeans_clustering
    elif method == 'nmf':
        clustering_method = nmf_clustering
    elif method == 'spectral':
        clustering_method = spectral_clustering

    # get centroids and assignment from clustering
    #
    if args.p2assignment != None:
        p2assign_file = args.p2assignment
        if os.path.isfile(p2assign_file):
            assignment = []
            with open(p2assign_file, 'r') as f:
                for line in f:
                    label = line.split()[0]
                    assignment.append(int(label))
            assignment = np.array(assignment)
        else:
            print('invalid assignment file, use clustering assignment.')
            D_final, assignment = clustering_method(X, k_cluster, numIter)
            acc, AMI = evaluation_clustering(assignment, Y)
            print('clustering accuracy = {:.4f}'.format(acc))
            print('AMI = {:.4f}'.format(AMI))
            print('D_final = ', D_final)
            print('shape of D mat:', D_final.shape)


    #### BEGIN of the regression part
    print('------ Begin the regression part...')
    df = pd.read_pickle(p2df_file)
    if expr_name == 'mouse' or expr_name == 'human_mgh':
        df = df.reset_index(drop=True)

    if expr_name == 'human_mgh':
        df = df.rename(columns = dict((k, k.replace("'", '')) for k in df.columns))
    print('df in regression shape = ', df.shape)


    # ipdb.set_trace()
    df_train, df_test, assign_train, assign_test = split_df_and_assignment(df, assignment)
    print('df test = ', df_test.shape)
    print('test data assignment set:', set(list(assign_test)))
    print('df train = ', df_train.shape)
    print('train data assignment set:', set(list(assign_train)))
    print('-' * 7)

    # p2tf_gene_list = os.path.join(gene_list_root, 'gene_id_TF_and_ensembl_pickle')
    if args.p2tf != None:
        p2tf_gene_list = args.p2tf
    else:
        p2tf_gene_list = os.path.join(gene_list_root, 'top_50_MAD_val_selected_TF_pickle')
    # p2tf_gene_list = os.path.join(gene_list_root, 'top_200_MAD_val_selected_TF_pickle')
    p2target_gene_list = os.path.join(gene_list_root, 'gene_id_non_TF_and_ensembl_pickle')


    with open(p2tf_gene_list, 'rb') as f:
        tf_list = pickle.load(f)
        tf_list = list(tf_list)
    with open(p2target_gene_list, 'rb') as f:
        target_list = pickle.load(f)
        target_list = list(target_list)

    if args.gene_list_type == 'symbol':
        with open('../data/merged_gene_id_to_name_pickle', 'rb') as f:
            ENSG_2_symbol_dict = pickle.load(f)
        # symbol_2_ENSG_dict = dict((v, k) for k, v in ENSG_2_symbol_dict.items())
        # tf_list = [ENSG_2_symbol_dict[a] for a in tf_list if a in ENSG_2_symbol_dict]
        target_list = [ENSG_2_symbol_dict[a] for a in target_list if a in ENSG_2_symbol_dict]

    query_target_list = get_top_k_non_zero_targets(num_target, df_train, target_list)
    print('-' * 7)

    print('.... generating train set')
    mat_dict_train, TF_ids, target_ids = load_mat_dict_and_ids(df_train, tf_list,
            query_target_list, assign_train, k_cluster)
    print('-' * 7)

    print('.... generating test set')
    mat_dict_test, _, _ = load_mat_dict_and_ids(df_test, tf_list, query_target_list,
            assign_test, k_cluster)
    print('-' * 7)

    # ### run cross_validation!!!!!! #############

    # print('start cross validation!!!')
    # list_of_l1 = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    # # list_of_l1 = [1e-1, 1e-2, 1e-3]
    # list_of_l2 = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5]
    # l1_opt, l2_opt, r2_opt = cross_validation(mat_dict_train, similarity, list_of_l1, list_of_l2)
    # print('cv done! lambda1 = {}, lambda2 = {}, opt R squared on eval {:.4f}'.format(
    #     l1_opt, l2_opt, r2_opt))
    # print('-' * 7)
    # lambda1 = l1_opt
    # lambda2 = l2_opt

    lambda1 = args.lambda1
    lambda2 = args.lambda2

    ############### end of cv ####################


    #### optimize using RCD
    # ipdb.set_trace()
    train_error, test_error = [], []
    r2_final_train, r2_final_test = [], []
    r2_final_0 = 0
    num_rep = 1
    for _ in range(num_rep):
        trained_weight_dict, weight_dict_0 = rcd_lasso_multi_cluster(mat_dict_train, similarity,
                                        lambda1, lambda2, slience = True)


        test_error.append(loss_function_value(mat_dict_test, trained_weight_dict, similarity,
                lambda1 = 0, lambda2 = 0))
        train_error.append(loss_function_value(mat_dict_train, trained_weight_dict, similarity,
                lambda1 = 0, lambda2 = 0))
        # ipdb.set_trace()
        r2_final_train.append(average_r2_score(mat_dict_train, trained_weight_dict))
        r2_final_test.append(average_r2_score(mat_dict_test, trained_weight_dict))
        r2_final_0 += average_r2_score(mat_dict_test, weight_dict_0)
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
    root_path = '/home/jianhao2/clus_GRN/data/'
    path_to_gene_name_dict = os.path.join(root_path, 'merged_gene_id_to_name_pickle')
    gene_id_name_mapping = load_dict(path_to_gene_name_dict)

    # for idx in range(k_cluster):
    #     print('cell type ', idx)
    #     find_top_k_TFs(5, query_target_list, trained_weight_dict[idx].T, TF_ids, gene_id_name_mapping)

    if expr_name != 'mouse' and expr_name != 'human_mgh':
        dict_to_saved = {'weight_dic' : trained_weight_dict,
                         'TF_ids'     : [gene_id_name_mapping[ids] for ids in TF_ids],
                         'query_targets' : [gene_id_name_mapping[ids] for ids in query_target_list]
                         }
    else:
        dict_to_saved = {'weight_dic' : trained_weight_dict,
                         'TF_ids'     : [symbols.upper() for symbols in TF_ids],
                         'query_targets' : [symbols.upper() for symbols in query_target_list]
                         }

    dict_name = '{}_data_{}_similarity_{}_target'.format(expr_dtype,
            similarity,
            num_target)
    # path_2_saved_dict = os.path.join('results_cima', dict_name)
    path_2_saved_dict = os.path.join('/data/jianhao/hepatocyte_update_dataset_101619', dict_name)
    save_dict(dict_to_saved, path_2_saved_dict)





    # -------------------
    # ###### VISUALIZATION
    # df_centroids = pd.DataFrame(D_final.reshape(k_cluster, len_of_gene), columns = feat_cols)
    # df_centroids['label'] = ['cell type {}'.format(x) for x in range(1, k_cluster + 1)]
    # # print('shape of centroid df:', df_centroids.shape)
    # # print('is D_centroids finite?', np.isfinite(df_centroids[feat_cols].values).all())

    # # # we need to normalize the input data X
    # df_final = pd.DataFrame(data = X, columns = feat_cols)
    # df_final['label'] = Y
    # df_final = df_final.append(df_centroids)
    # print('shape of df_final: ', df_final.shape)
    # #
    # # # # run tSNE for visualization
    # # tmp = '{gene_set}_smoothed_{raw_bool}_{n_cells}_cells_{cmethod}_magic'.format(
    # #         gene_set = set_of_gene, raw_bool = raw_data,
    # #         n_cells = cells_per_cluster, cmethod = method)
    # # file_name_fig = tmp
    # # p2f = os.path.join(data_root, 'pic', file_name_fig)


    # tmp = '{gene_set} smoothed: {data_type}, {n_cells} cells {cmethod}'.format(
    #         gene_set = set_of_gene, data_type = expr_dtype,
    #         n_cells = cells_per_cluster, cmethod = method)
    # fig_title = '{}\n acc: {:.4f}, AMI: {:.4f}'.format(tmp, acc, AMI)

    # assign_label = ['cell type {}'.format(n) for n in assignment]
    # tsne_df_refine(df_final, feat_cols, k_cluster, assignment, Y, '../tmp_feb.png', fig_title)
    # # # tsne_df(df_final, feat_cols, cells_per_cluster, k_cluster, p2f, fig_title)
