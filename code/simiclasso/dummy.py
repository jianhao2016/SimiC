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
import time
# 
p2df = '/data/jianhao/hepatocyte_update_dataset_101619/magic_cell_mat_w_label_pd'
p2assignment = '/data/jianhao/hepatocyte_update_dataset_101619/monoc_simple_group_assign'
k_cluster = 3
similarity = True
p2tf = '/data/jianhao/hepatocyte_update_dataset_101619/top_200_MAD_val_selected_TF_pickle'
p2saved_file = '/data/jianhao/hepatocyte_update_dataset_101619/test_time_output/3_states_res_dict_simic'
num_TFs = 200
num_target_genes = 1000
max_rcd_iter = 50000
df_with_label = True
lambda1=1e-5
lambda2=0.1
_NF = 100

ts_simic = time.time()
simicLASSO_op(p2df, p2assignment, k_cluster, similarity, p2tf, p2saved_file, num_TFs, num_target_genes, _NF = _NF, 
        max_rcd_iter = max_rcd_iter, df_with_label = df_with_label,
        lambda1=lambda1, lambda2 = lambda2)
te_simic = time.time()
t_simic = te_simic - ts_simic 

time_pass = lambda x: '{}h{}min'.format(x // 3600, x// 60 - x//3600)
print('simic uses {}'.format(time_pass(t_simic)))

ts_auc = time.time()
p2AUC = '/data/jianhao/hepatocyte_update_dataset_101619/test_time_output/3_states_AUC_simic'
main_fn(p2df, p2saved_file, p2AUC)
te_auc = time.time()

t_auc = te_auc - ts_auc
t_total = te_auc - ts_simic


print('simic uses {}'.format(time_pass(t_simic)))
print('auc uses {}'.format(time_pass(t_auc)))
print('total uses {}'.format(time_pass(t_total)))


# p2df = '/data/jianhao/hepatocyte_update_dataset_101619/Bee_DF.pickle'
# p2assignment = '/data/jianhao/hepatocyte_update_dataset_101619/Bees_assignment'
# k_cluster = 2
# similarity = True
# # p2tf = '/data/jianhao/mouse_TF.pickle'
# p2tf = '/data/jianhao/hepatocyte_update_dataset_101619/Bee_TFs.pickle'
# p2saved_file = '/data/jianhao/hepatocyte_update_dataset_101619/Bees_results_update_1117'
# num_TFs = 200
# num_target_genes = 2000
# max_rcd_iter = 500000
# df_with_label = False
# _NF = 100
# 
# # simicLASSO_op(p2df, p2assignment, k_cluster, similarity, p2tf, p2saved_file, num_TFs, num_target_genes, _NF = _NF, 
# #         max_rcd_iter = max_rcd_iter, df_with_label = df_with_label)
# 
# p2AUC = '/data/jianhao/hepatocyte_update_dataset_101619/AUC_dict_monoc_3groups'
# main_fn(p2df, p2saved_file, p2AUC)
# 
