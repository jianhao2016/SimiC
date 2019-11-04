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

def magic_op_on_df(p2df, p2fc, p2save_df):
    '''
    read a pandas dataframe, performs magic on selected feature columns,
    Return an imputed dataframe.
    '''
    # root = '/data/jianhao/clus_GRN/'
    # df_path = os.path.join(root, 'raw_expression_data', 'pandas_dataframe_mouse')
    if os.path.isfile(p2df):
        df_path = p2df
    else:
        raise ValueError('{} is not a valid dataframe'.format(p2df))
    tmp = pd.read_pickle(df_path)
    # tmp = tmp.to_sparse()
    # print(type(tmp))
    # print(tmp)
    #tmp = scprep.filter.remove_empty_genes(tmp)
    if os.path.isfile(p2fc):
        fc_path = p2fc
    else:
        raise ValueError('{} is not a valid feature columns'.format(p2fc))
    # fc_path = os.path.join(root, 'diff_gene_list', 'df_feature_column_mouse')
    with open(fc_path, 'rb') as f:
        feat_col = pickle.load(f)
    X = tmp[feat_col]
    # in case there are na values in dataframe, fill it with zero.
    X.fillna(0, inplace = True)
    
    magic_op = magic.MAGIC()
    X_magic = magic_op.fit_transform(X)
    
    
    X_magic['label'] = tmp['label']
    print('shape of X_magic:', X_magic.shape)
    # X_magic.to_pickle(os.path.join(root, 'pandas_dataframe_mouse'))
    print('saving imputed dataframe to: ', p2save_df)
    X_magic.to_pickle(p2save_df)
