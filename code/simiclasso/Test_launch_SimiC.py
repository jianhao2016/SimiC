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
from filter_Ws import filter_Ws
import time
# 
 

# simic_val_Granja
# simic_val_Granja_filtered

p2df = '/home/sevastopol/data/gserranos/SimiC/Dev_SimiC/simicLASSO_git/data/CIMA_test/simic_val_Kalsotra_QTP.DF.pickle'
p2assignment = '/home/sevastopol/data/gserranos/SimiC/Dev_SimiC/simicLASSO_git/data/CIMA_test/simic_val_Kalsotra_QTP.clustAssign.txt'
p2tf = '/home/sevastopol/data/gserranos/SimiC/Dev_SimiC/simicLASSO_git/data/CIMA_test/simic_val_Kalsotra_QTP.TFs.pickle'

similarity = True
k_cluster = None
num_TFs = -1
num_target_genes = -1
max_rcd_iter = 100000
df_with_label = False
lambda1=0.01
lambda2=0.01
percent_of_target = 1
cross_val = False
cross_val = True


# p2saved_file = '/home/sevastopol/data/gserranos/SimiC/Dev_SimiC/simicLASSO_git/data/CIMA_test/simic_val_Granja_filtered_L1'+str(lambda1)+'_L2'+str(lambda2)+'_Ws_filtered_BIC.pickle' #Weigths
# p2AUC = '/home/sevastopol/data/gserranos/SimiC/Dev_SimiC/simicLASSO_git/data/CIMA_test/simic_val_Granja_filtered_L1'+str(lambda1)+'_L2'+str(lambda2)+'_AUCs_Scaled_filtered_BIC.pickle' #Auc


lambda1= 0.01
lambda2_list =[10, 0.1, 0.01, 0.001, 1e-5, 0]
lambda2_list = [0.01]
if cross_val:
	print('CROSS VALIDATION')
	lambda1='CrossVal'
	lambda2='CrossVal'
	p2saved_file = '/home/sevastopol/data/gserranos/SimiC/Dev_SimiC/simicLASSO_git/data/CIMA_test/Test_L1'+str(lambda1)+'_L2'+str(lambda2)+'_Ws_filtered_BIC.pickle' #Weigths
	p2AUC = '/home/sevastopol/data/gserranos/SimiC/Dev_SimiC/simicLASSO_git/data/CIMA_test/Test_L1'+str(lambda1)+'_L2'+str(lambda2)+'_AUCs_Scaled_filtered_BIC.pickle' #Auc
	print('CROSS VALIDATION')
	simicLASSO_op(p2df, p2assignment, similarity, p2tf, p2saved_file,  k_cluster, num_TFs, num_target_genes, 
			max_rcd_iter = max_rcd_iter, df_with_label = df_with_label,
			cross_val=cross_val)

else:
	for lambda2 in lambda2_list:

		ts_simic = time.time()

		# CROSS VALIDATION 
		# SET LAMBDA1 AND LAMBDA2

		print('\n\nSET LAMBDA1 AND LAMBDA2\n\n')
		simicLASSO_op(p2df, p2assignment, similarity, p2tf, p2saved_file,  k_cluster, num_TFs, num_target_genes, 
				max_rcd_iter = max_rcd_iter, df_with_label = df_with_label,
				lambda1=lambda1, lambda2 = lambda2)
		p2saved_file = '/home/sevastopol/data/gserranos/SimiC/Dev_SimiC/simicLASSO_git/data/CIMA_test/simic_val_Kalsotra_Time_L10.001_L20.001_Ws.pickle'
		filter_Ws(p2saved_file, plot_histogram = False, path2plot = None)
		te_simic = time.time()
		t_simic = te_simic - ts_simic 

		time_pass = lambda x: '{}h{}min'.format(x // 3600, x// 60 - x//3600)
		print('simic uses {}'.format(time_pass(t_simic)))

		# ts_auc = time.time()
		# # main_fn(p2df, p2saved_file, p2AUC, percent_of_target = percent_of_target)
		# te_auc = time.time()

		# t_auc = te_auc - ts_auc
		# t_total = te_auc - ts_simic


		# print('simic uses {}'.format(time_pass(t_simic)))
		# print('auc uses {}'.format(time_pass(t_auc)))
		# print('total uses {}'.format(time_pass(t_total)))

print('Yay!')
