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
import pdb
import os
import argparse
import ipdb
from sklearn.linear_model import LassoLars
from sklearn.preprocessing import normalize
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import NMF
from sklearn.metrics import accuracy_score, adjusted_mutual_info_score
from visualization import tsne_df
from common_io import load_dataFrame

def kmeans_clustering(X, k_cluster, numIter):
    kmeans = KMeans(n_clusters = k_cluster, max_iter = numIter)
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_
    assignment = kmeans.labels_
    return centroids, assignment

def nmf_clustering(X, k_cluster, numIter):
    nmf_model = NMF(n_components = k_cluster, solver = 'mu', max_iter = numIter, alpha = 0)

    weight_matrix = nmf_model.fit_transform(X)
    centroid_matrix = nmf_model.components_

    _, assignment = kmeans_clustering(weight_matrix, k_cluster, numIter)

    return centroid_matrix, assignment

def spectral_clustering(X, k_cluster, numIter):
    spec_clustering = SpectralClustering(n_clusters = k_cluster, assign_labels = 'discretize')
    spec_clustering.fit(X)
    assignment = spec_clustering.labels_

    all_labels = list(set(assignment))
    clusters_dict = {i:[] for i in all_labels}

    for idx in range(len(assignment)):
        sample_label = assignment[idx]
        if len(clusters_dict[sample_label]) == 0:
            clusters_dict[sample_label] = X[idx]
        else:
            clusters_dict[sample_label] = np.vstack((clusters_dict[sample_label], X[idx]))

    # print(clusters_dict[0])
    # print(clusters_dict[0].shape)
    centroids = np.array([np.mean(clusters_dict[idx], axis = 0) for idx in all_labels])
    return centroids, assignment

def evaluation_clustering(pred_label, true_label):
    # both input should be 1d array
    all_predict_label = list(set(pred_label))
    clusters_label_distn = {i : [] for i in all_predict_label}
    for pl, tl in zip(pred_label, true_label):
        if len(clusters_label_distn[pl]) == 0:
            clusters_label_distn[pl] = np.array([tl])
        else:
            clusters_label_distn[pl] = np.concatenate((clusters_label_distn[pl], np.array([tl])))

    for pl in clusters_label_distn:
        uniq, idx = np.unique(clusters_label_distn[pl], return_inverse=True)
        most_frequent_label = uniq[np.argmax(np.bincount(idx))]
        clusters_label_distn[pl] = most_frequent_label

    pred_label_str = np.array(list(map(lambda x: clusters_label_distn[x], pred_label)))
    # print(pred_label_str)
    all_true_label_list = list(set(true_label))
    label_str_2_int = {}
    for idx in range(len(all_true_label_list)):
        label_str_2_int[all_true_label_list[idx]] = idx
    pred_label_int = np.array(list(map(lambda x: label_str_2_int[x], pred_label_str)))
    true_label_int = np.array(list(map(lambda x: label_str_2_int[x], true_label)))

    acc = accuracy_score(true_label_int, pred_label_int)
    AMI = adjusted_mutual_info_score(true_label_int, pred_label_int, average_method='arithmetic')
    return acc, AMI


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--numIter', type=int, default=1000)
    parser.add_argument('--method', type=str, choices=['kmeans', 'nmf', 'spectral'], 
            default = 'kmeans')
    parser.add_argument('--NF', type=float, default=1, help='normalization factor')
    args = parser.parse_args()

    # set number of iteration, lambda in lasso, epsilon in dictionary update and normalization factor
    print(args)
    numIter = args.numIter
    method = args.method
    _NF = args.NF

    data_root = '/data/jianhao/clus_GRN'
    k_cluster = 3
    cells_per_cluster = 500
    raw_data = False
    set_of_gene = 'not_encode_v29'

    if raw_data:
        p2df_file = os.path.join(data_root, 'raw_expression_data', 'pandas_dataframe_{}'.format(cells_per_cluster))
    else:
        # file_name_df = 'smoothed_dataframe_{}_cells_per_cluster_all_gene_coexpression'.format(cells_per_cluster)
        file_name_df = os.path.join(data_root, 'magic_smoothed_all_genes_500_cells_per_cluster')
        p2df_file = os.path.join(data_root, 'ensembl94', file_name_df)

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

    p2feat_file = os.path.join(data_root, 'diff_gene_list', file_name_feat_cols)


    with open(p2feat_file, 'rb') as f:
        original_feat_cols = pickle.load(f)
    # ipdb.set_trace()

    # tmp_df = pd.read_pickle(os.path.join(data_root, 'ensembl94', 'smoothed_dataframe_100_cells_per_cluster_all_gene_coexpression'))
    # ensembl_list = list(tmp_df.columns.values)

    # intersection = set.intersection(set(original_feat_cols), set(ensembl_list))
    # original_feat_cols = list(intersection)

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
    D_final, assignment = clustering_method(X, k_cluster, numIter)
    acc, AMI = evaluation_clustering(assignment, Y)
    print('clustering accuracy = {:.4f}'.format(acc))
    print('AMI = {:.4f}'.format(AMI))
    print('D_final = ', D_final)
    print('shape of D mat:', D_final.shape)

    df_centroids = pd.DataFrame(D_final.reshape(k_cluster, len_of_gene), columns = feat_cols)
    df_centroids['label'] = ['cell type {}'.format(x) for x in range(1, k_cluster + 1)]
    print('shape of centroid df:', df_centroids.shape)
    print('is D_centroids finite?', np.isfinite(df_centroids[feat_cols].values).all())

    # we need to normalize the input data X
    # df_final = df.copy()
    df_final = pd.DataFrame(data = X, columns = feat_cols)
    df_final['label'] = Y
    # df_final[feat_cols] = normalize(X_raw)
    # df_final[feat_cols] = X
    df_final = df_final.append(df_centroids)
    print('shape of df_final: ', df_final.shape)
    
    # run tSNE for visualization
    tmp = '{gene_set}_smoothed_{raw_bool}_{n_cells}_cells_{cmethod}_magic'.format(
            gene_set = set_of_gene, raw_bool = raw_data, 
            n_cells = cells_per_cluster, cmethod = method)
    file_name_fig = tmp
    p2f = os.path.join(data_root, 'pic', file_name_fig)


    tmp = '{gene_set} smoothed: {raw_bool}, {n_cells} cells {cmethod}'.format(
            gene_set = set_of_gene, raw_bool = raw_data, 
            n_cells = cells_per_cluster, cmethod = method)
    fig_title = '{}\n acc: {:.4f}, AMI: {:.4f}'.format(tmp, acc, AMI)

    tsne_df(df_final, feat_cols, cells_per_cluster, k_cluster, p2f, fig_title)
    
