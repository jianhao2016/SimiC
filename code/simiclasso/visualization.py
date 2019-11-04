#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 jianhao2 <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
This script is only for visualization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.manifold import TSNE

def tsne_df(df_final, feat_cols, cells_per_cluster, k_cluster, p2f, fig_title):
    '''
    input:
        a dataframe with label
        p2f: path to saved fig

    output:
        None
    '''
    rndperm = np.random.permutation(df_final.shape[0])
    n_sne = k_cluster * cells_per_cluster + k_cluster

    t_start = time.time()
    tsne = TSNE(n_components = 2, verbose = 1, perplexity=40, n_iter = 1000, random_state = 42)
    tsne_result = tsne.fit_transform(df_final[feat_cols].values)
    print('tsne_result shape = ', tsne_result.shape)

    print('t-SNE done! Time elapse {:.04f}s'.format(time.time() - t_start))
    df_tsne = df_final.copy()
    df_tsne['x-tsne'] = tsne_result[:, 0]
    df_tsne['y-tsne'] = tsne_result[:, 1]

    sc_x = df_tsne['x-tsne'].values[:k_cluster * cells_per_cluster]
    sc_y = df_tsne['y-tsne'].values[:k_cluster * cells_per_cluster]
    sc_types = df_tsne['label'].values[:k_cluster * cells_per_cluster]

    centroid_x = df_tsne['x-tsne'].values[-k_cluster:]
    centroid_y = df_tsne['y-tsne'].values[-k_cluster:]
    centroid_types = df_tsne['label'].values[-k_cluster:]
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    color_list = ['red', 'green', 'blue']

    for idx in range(3):
        ax.scatter(sc_x[idx*cells_per_cluster:(idx+1)*cells_per_cluster], 
                sc_y[idx*cells_per_cluster:(idx+1)*cells_per_cluster], 
                color = color_list[idx], label = sc_types[idx * cells_per_cluster], 
                s = 10, alpha = 0.6)

    marker_list = ['D', 's', '+']
    centroid_color_list = ['yellow', 'black', 'magenta']
    for idx in range(k_cluster):
        ax.scatter(centroid_x[idx], centroid_y[idx], color = centroid_color_list[idx], marker = marker_list[idx],
                alpha = 0.8, label = centroid_types[idx])
    
    ax.legend()
    ax.set_title(fig_title)
    ax.set_xlabel('tSNE element 1')
    ax.set_ylabel('tSNE element 2')

    ax.grid(color = 'grey', alpha = 0.4)
    # p2f = os.path.join(data_root, 'pic/ensembl94', 'coexpression_{}_based_clustering_lm_gene_500_non_smooth'.format(method))
    fig.savefig(p2f, dpi = 500)

def get_cmap(n, name='gist_rainbow'):
    '''
    Returen a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color. name must be a standard mpl colormap name.
    '''
    color_map = plt.cm.get_cmap(name, n)
    rgb_list = [color_map(idx) for idx in range(n)]
    return rgb_list

def tsne_df_refine(df_final, feat_cols, k_cluster, assignment, Y, p2f, fig_title):
    '''
    input:
        a dataframe with label
        p2f: path to saved fig
        assignment: from clustering
        Y: true label

    output:
        None
    '''
    t_start = time.time()
    tsne = TSNE(n_components = 2, verbose = 1, perplexity=40, n_iter = 1000, random_state = 42)
    tsne_result = tsne.fit_transform(df_final[feat_cols].values)
    print('tsne_result shape = ', tsne_result.shape)

    print('t-SNE done! Time elapse {:.04f}s'.format(time.time() - t_start))
    df_tsne = df_final.copy()
    df_tsne['x-tsne'] = tsne_result[:, 0]
    df_tsne['y-tsne'] = tsne_result[:, 1]

    data_len = len(Y)
    assert len(Y) == len(assignment)
    sc_x = df_tsne['x-tsne'].values[:data_len]
    sc_y = df_tsne['y-tsne'].values[:data_len]
    # sc_types = df_tsne['label'].values[:k_cluster * cells_per_cluster]

    centroid_x = df_tsne['x-tsne'].values[-k_cluster:]
    centroid_y = df_tsne['y-tsne'].values[-k_cluster:]
    centroid_types = df_tsne['label'].values[-k_cluster:]
    
    # fig = plt.figure()
    fig, axes = plt.subplots(nrows = 1, ncols = 2, 
            sharey = True, dpi = 150)
    # color_list = ['red', 'green', 'blue']
    color_list = get_cmap(k_cluster)

    label_list = [assignment, Y]
    title_list = ['clustering assigned label', 'true label']
    for idx, cur_label in enumerate(label_list):
        # cur_label = label_list[idx]
        label_set = list(set(cur_label))
        cur_title = title_list[idx]
        for color_idx, label in enumerate(label_set):
            color = color_list[color_idx]
            data_idx = [n for n,tmp_x in enumerate(cur_label) if tmp_x == label]
            axes[idx].scatter(sc_x[data_idx], sc_y[data_idx],
                    color = color, label = label,
                    s = 10, alpha = 0.6)
            axes[idx].grid(color = 'grey', alpha = 0.4)
            axes[idx].set_title(cur_title)
            axes[idx].set_xlabel('tSNE element 1')
            axes[idx].set_ylabel('tSNE element 2')
            axes[idx].legend()

    # for idx in range(3):
    #     ax.scatter(sc_x[idx*cells_per_cluster:(idx+1)*cells_per_cluster], 
    #             sc_y[idx*cells_per_cluster:(idx+1)*cells_per_cluster], 
    #             color = color_list[idx], label = sc_types[idx * cells_per_cluster], 
    #             s = 10, alpha = 0.6)

    # marker_list = ['D', 's', '+']
    centroid_color_list = ['black'] * k_cluster
    for idx in range(k_cluster):
        axes[0].scatter(centroid_x[idx], centroid_y[idx], color = centroid_color_list[idx], 
                marker = 'D',
                alpha = 0.8, label = centroid_types[idx])
    

    plt.suptitle(fig_title)
    plt.tight_layout(rect=[0, 0.05, 1, 0.9])
    # p2f = os.path.join(data_root, 'pic/ensembl94', 'coexpression_{}_based_clustering_lm_gene_500_non_smooth'.format(method))
    fig.savefig(p2f, dpi = 150)


