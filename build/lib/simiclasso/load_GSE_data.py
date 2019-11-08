#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 qizai <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
This script is to load the data from GSE
"""

import csv
import os
import scipy.io
import numpy as np
import pandas as pd
import pickle
from scipy.sparse import coo_matrix, hstack

# genome = 'hg19'
# cell_lists = ['cd19_b_cells', 'cd56_natural_killer_cells', 'regulatory_t_cells']
# root_dir = '/Users/qizai/projects/NMF'
# human_matirces_dir = [os.path.join(root_dir, p, genome) for p in cell_lists]
# 
# num_cells_each_cluster = 100
# shrinked_mat_list = []
# cell_type_list = []
# gene_list = []
# 
# for data_path, cell_type in zip(human_matirces_dir, cell_lists):
#     if os.path.isdir(data_path):
#         mat_tmp = scipy.io.mmread(os.path.join(data_path, 'matrix.mtx'))
#         mat_tmp = mat_tmp.tocsc()
#         shrinked_mat = mat_tmp[:, :num_cells_each_cluster]
#         print('sum of first 5 cells:', shrinked_mat[:, :5].sum(axis = 0))
#         shrinked_mat_list.append(shrinked_mat)
# 
#         gene_path_tmp = os.path.join(data_path, 'genes.tsv')
#         gene_ids_tmp = [row[0] for row in csv.reader(open(gene_path_tmp), delimiter='\t')]
#         print('number of genes being measure:', len(gene_ids_tmp))
#         gene_list = gene_ids_tmp
# 
#         barcodes_path = os.path.join(data_path, 'barcodes.tsv')
#         barcodes_tmp = [row[0] for row in csv.reader(open(barcodes_path), delimiter='\t')]
#         shrinked_barcodes_tmp = barcodes_tmp[:num_cells_each_cluster]
#         print('number of cells in cluster: ', len(shrinked_barcodes_tmp))
#         cell_type_list += [cell_type] * len(shrinked_barcodes_tmp)
# 
#     else:
#         print('{} is not a valid path!'.format(data_path))
#         #print(sum(mat_tmp.toarray()))
data_file = '/data/jianhao/clus_GRN/human_MGH_GSE_dataset/GSE70630_OG_processed_data_v2.txt'
df = pd.read_table(data_file, index_col = 0)
df = df.transpose()
gene_list = list(df.columns.values)

# dense_mat = hstack(shrinked_mat_list).todense()
# print('size of stack dense mat: ', dense_mat.shape)
# np.save('dense_data', dense_mat)
# X = dense_mat.T
# df = pd.DataFrame(data = X, columns = gene_ids_tmp)
with open('/data/jianhao/clus_GRN/diff_gene_list/df_feature_column_human_MGH', 'wb') as f:
    pickle.dump(gene_list, f)

cell_type_list = [a.split('_')[0] for a in df.index.values]

df['label'] = cell_type_list
wrong_ids = [a for a in df.index.values if a[:3] != 'MGH']
df = df.drop(wrong_ids)

print('Size of data frame', df.shape)
print(df['label'].shape)
df.to_pickle('/data/jianhao/clus_GRN/raw_expression_data/pandas_dataframe_human_MGH')
