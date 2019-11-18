#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 jianhao2 <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
test file for simicLASSO_op
"""

from clus_regression import simicLASSO_op
from weighted_AUC_mat import main_fn

p2df = '/data/jianhao/hepatocyte_update_dataset_101619/Bee_DF.pickle'
p2assignment = '/data/jianhao/hepatocyte_update_dataset_101619/Bees_assignment'
k_cluster = 2
similarity = True
# p2tf = '/data/jianhao/mouse_TF.pickle'
p2tf = '/data/jianhao/hepatocyte_update_dataset_101619/Bee_TFs.pickle'
p2saved_file = '/data/jianhao/hepatocyte_update_dataset_101619/Bees_results_update_1117'
num_TFs = 200
num_target_genes = 2000
max_rcd_iter = 500000
df_with_label = False
_NF = 100

simicLASSO_op(p2df, p2assignment, k_cluster, similarity, p2tf, p2saved_file, num_TFs, num_target_genes, _NF = _NF, 
        max_rcd_iter = max_rcd_iter, df_with_label = df_with_label)

p2AUC = '/data/jiahao/hepatocyte_update_dataset_101619/AUC_dict_monoc_3groups'
main_fn(p2df, p2saved_file, p2AUC)

