#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 jianhao2 <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
run magic on  cima dataset
"""
import pandas as pd
import pickle
import magic
import ipdb
import scprep
import os

root = '/data/jianhao/clus_GRN/'
df_path = os.path.join(root, 'raw_expression_data', 'pandas_dataframe_mouse')
tmp = pd.read_pickle(df_path)
# tmp = tmp.to_sparse()
# print(type(tmp))
# print(tmp)
#tmp = scprep.filter.remove_empty_genes(tmp)
fc_path = os.path.join(root, 'diff_gene_list', 'df_feature_column_mouse')
with open(fc_path, 'rb') as f:
    feat_col = pickle.load(f)
X = tmp[feat_col]
# in case there are na values in dataframe, fill it with zero.
X.fillna(0, inplace = True)

ipdb.set_trace()

magic_op = magic.MAGIC()
X_magic = magic_op.fit_transform(X)


X_magic['label'] = tmp['label']
print('shape of X_magic:', X_magic.shape)
X_magic.to_pickle(os.path.join(root, 'pandas_dataframe_mouse'))
