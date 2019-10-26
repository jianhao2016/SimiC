#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 jianhao2 <jianhao2@illinois.edu>
#
# Distributed under terms of the MIT license.

"""
This script will take the gene.tsv from 10xGenomics as input, 
extract a {gene_id: gene name} dictionary, save in folder
and return a union of dictionary.
"""

import pickle
import os
import csv

def from_tsv_to_dict(path_2_file):
    '''
    take a tsv file from 10xGenomics as input
    return a dictionary:
        {gene id: gene name}
    '''
    reader = csv.reader(open(path_2_file), delimiter = '\t')
    id2name_dict = {}
    error_id_list = []
    num_genes = 0
    for [gene_id, gene_name] in reader:
        if gene_id not in id2name_dict:
            id2name_dict[gene_id] = gene_name
            num_genes += 1
        elif id2name_dict[gene_id] != gene_name:
            error_id_list.append(gene_id)
            print('id - name mismatch at:', gene_id, gene_name)

    print('number of genes in tsv file:', num_genes)
    return id2name_dict, error_id_list

def cmp_2_dictionary(dict1, dict2):
    '''
    take two dictionaries as input, 
    compare the intersection part, see if same keys have same values
    '''
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())
    same_keys = keys1.intersection(keys2)

    diff_dict = {}
    for gene_id in same_keys:
        name_1 = dict1[gene_id]
        name_2 = dict2[gene_id]
        if name_1 != name_2:
            diff_dict[gene_id] = { 'dict1': name_1, 
                                    'dict2': name_2}
            print('gene name different!', gene_id)
            print(name_1, '\t', name_2)

    return diff_dict


def merge_2_dictionary(dict1, dict2):
    '''
    take two dictionaries as input, 
    return the union of them.
    '''
    diff_dict = cmp_2_dictionary(dict1, dict2)
    if len(diff_dict) != 0:
        print('dict1 and dict2 have different values, use dict2 as default')
        print('---')
        print('number of mismatched genes = ', len(diff_dict))
    union_dict = {**dict1, **dict2}

    print('merged dictionary length =', len(union_dict))

    return union_dict

def save_dict(d, path_2_file):
    print('dumping the dictionary to disk in pickle...')
    with open(path_2_file, 'wb') as f:
        pickle.dump(d, f)
        print('Done dumping!')

def load_dict(path_2_file):
    print('loading pickle dictionary from file...')
    with open(path_2_file, 'rb') as f:
        tmp = pickle.load(f)
        print('Done loading!')
    return tmp
    
if __name__ == '__main__':
    hg_gene_file = 'hg_genes.tsv'
    path_2_hg_genes = os.path.join('/home/jianhao2/clus_GRN/data', hg_gene_file)

    young_type = 'Young1'
    cima_gene_file = 'genes.tsv'
    path_2_cima_genes = os.path.join('/data/cima', young_type, 'GRCh38', cima_gene_file)

    dict_hg, _ = from_tsv_to_dict(path_2_hg_genes)
    save_dict(dict_hg, 'hg_gene_id_to_name_pickle')
    print('-' * 7)

    dict_young, _ = from_tsv_to_dict(path_2_cima_genes)
    save_dict(dict_young, 'Young_gene_id_to_name_pickle')
    print('-' * 7)

    merge_dict = merge_2_dictionary(dict_hg, dict_young)
    save_dict(merge_dict, 'merged_gene_id_to_name_pickle')
    print('-' * 7)


