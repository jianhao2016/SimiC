#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qizai <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
This script is to load the 10x Genomics data
"""

import csv
import os
import scipy.io
import numpy as np
import pandas as pd
import pickle
from scipy.sparse import coo_matrix, hstack

genome = 'hg19'
cell_lists = ['cd19_b_cells', 'cd56_natural_killer_cells', 'regulatory_t_cells']
root_dir = '/Users/qizai/projects/NMF'
human_matirces_dir = [os.path.join(root_dir, p, genome) for p in cell_lists]

num_cells_each_cluster = 100
shrinked_mat_list = []
cell_type_list = []
gene_list = []

for data_path, cell_type in zip(human_matirces_dir, cell_lists):
    if os.path.isdir(data_path):
        mat_tmp = scipy.io.mmread(os.path.join(data_path, 'matrix.mtx'))
        mat_tmp = mat_tmp.tocsc()
        shrinked_mat = mat_tmp[:, :num_cells_each_cluster]
        print('sum of first 5 cells:', shrinked_mat[:, :5].sum(axis = 0))
        shrinked_mat_list.append(shrinked_mat)

        gene_path_tmp = os.path.join(data_path, 'genes.tsv')
        gene_ids_tmp = [row[0] for row in csv.reader(open(gene_path_tmp), delimiter='\t')]
        print('number of genes being measure:', len(gene_ids_tmp))
        gene_list = gene_ids_tmp

        barcodes_path = os.path.join(data_path, 'barcodes.tsv')
        barcodes_tmp = [row[0] for row in csv.reader(open(barcodes_path), delimiter='\t')]
        shrinked_barcodes_tmp = barcodes_tmp[:num_cells_each_cluster]
        print('number of cells in cluster: ', len(shrinked_barcodes_tmp))
        cell_type_list += [cell_type] * len(shrinked_barcodes_tmp)

    else:
        print('{} is not a valid path!'.format(data_path))
        #print(sum(mat_tmp.toarray()))

dense_mat = hstack(shrinked_mat_list).todense()
print('size of stack dense mat: ', dense_mat.shape)
np.save('dense_data', dense_mat)
X = dense_mat.T
df = pd.DataFrame(data = X, columns = gene_ids_tmp)
with open('df_feature_column', 'wb') as f:
    pickle.dump(gene_list, f)

df['label'] = cell_type_list
print('Size of data frame', df.shape)
print(df['label'].shape)
print(df['label'][:10], df['label'][100:110], df['label'][200:210])
df.to_pickle('pandas_dataframe')
