#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 jianhao2 <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
This script contains various evaluation metrics and functions
dealing with DataFrame matrix
"""
from sklearn.metrics import r2_score
import requests
import statistics
import numpy as np

def get_top_k_idx_from_list(query_list, top_k):
    if not isinstance(query_list, np.ndarray):
        tmp_list = np.array(query_list)
    else:
        tmp_list = query_list
    idx_list = np.argpartition(tmp_list, -top_k)
    top_k_idx = idx_list[-top_k:]
    return top_k_idx

def get_r_squared(y_true, y_pred, k = 0,  
        sample_weight = None,
        multioutput='uniform_average'):
    '''
    compute R2 score for regression model,
    if y is a matirx, (multiple y_vector), then first compute R2 score for
    each column, and get average of column R2 score.
    otherwise, see multioutput explaination in sklearn.
    k: number of independent variable in X
        k = 0, ordinary R2, nothing changed
        k = X.shape[1], Adjusted R-Squared, 
    '''
    R2_val = r2_score(y_true, y_pred, sample_weight, multioutput = multioutput)

    # num_sample = len(y_true)
    num_sample = y_pred.shape[0]
    assert num_sample > 1
    adj_R2 = 1 - (1 - R2_val) * ((num_sample - 1)/(num_sample - k - 1))
    return adj_R2

def Median_Abs_Deviation(x_array):
    '''
    find the meidan of deviation from median of x_array
    '''
    abs_dev = abs(x_array - statistics.median(x_array))
    MAD = statistics.median(abs_dev)
    return MAD


def get_chr_postion(gene_ids):
    '''
    get chromosome position in bed format from MyGene.info
    using batch POST
    return a list of query results.
    '''
    target_gene_str = ','.join(gene_ids)
    headers = {'content-type': 'application/x-www-form-urlencoded'}
    species = 'human'
    scopes = 'ensembl.gene'
    fields = 'symbbol,genomic_pos_hg19.chr, genomic_pos_hg19.start, genomic_pos_hg19.end'
    params = 'q={}&species={}&scopes={}&fields={}'.format(target_gene_str,
                                                        species, scopes, 
                                                        fields)
    res = requests.post('http://mygene.info/v3/query', data=params, headers=headers)
    r_data = res.json()
    return r_data

def get_BED_from_request(r_data, expand_bp = 1000):
    '''
    extract 'chX start end' from r_data
    saved in a dict { ensembl_id: position_str}
    '''
    gene_id_to_position = {}
    ch_set = set([str(i) for i in range(23)] + ['X', 'Y'])
    bed_str = ''
    for query_item in r_data:
        gene_id = query_item['query']
        g_pos = query_item['genomic_pos_hg19']
        if isinstance(g_pos, list):
            for tmp in g_pos:
                if tmp['chr'] in ch_set: 
                    break
            g_pos = tmp
        assert isinstance(g_pos, dict)
        chX = g_pos['chr']
        start_pos = max(g_pos['start'] - expand_bp, 0)
        end_pos = g_pos['end'] + expand_bp
        pos_str = 'ch{}\t{}\t{}'.format(chX, start_pos, end_pos)
        gene_id_to_position[gene_id] = pos_str
        bed_str += pos_str + '\n'
    print(bed_str)
    return gene_id_to_position

if __name__ == '__main__':
    gene_ids = ['ENSG00000231500', 'ENSG00000177954', 'ENSG00000142541', 'ENSG00000147403']
    r_data = get_chr_postion(gene_ids)
    gene_id_to_position = get_BED_from_request(r_data)
    # print(gene_id_to_position)

    a = np.array([1, 1, 2, 2, 4, 6, 9])
    mad = Median_Abs_Deviation(a)
    print(mad)
    b = np.random.permutation(a)
    print(b)
    idx = get_top_k_idx_from_list(b, 4)
    print(idx)
    print(b[idx])



