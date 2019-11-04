#! /bin/sh
#
# dummy.sh
# Copyright (C) 2019 jianhao2 <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.
#

python clus_regression.py --p2df $hepa_df --p2fc $hepa_fc --p2assignment $hepa_ag --k_cluster 5 --similarity 1 --p2tf /data/jianhao/hepatocyte_update_dataset_101619/top_50_MAD_val_selected_TF_pickle --num_target_genes 250 --gene_list_type symbol

python clus_regression.py --p2df $hepa_df --p2fc $hepa_fc --p2assignment $hepa_ag --k_cluster 5 --similarity 1 --p2tf /data/jianhao/hepatocyte_update_dataset_101619/top_200_MAD_val_selected_TF_pickle --num_target_genes 1000 --gene_list_type symbol
