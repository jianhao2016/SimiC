import os
import pickle
import math as m
import numpy as np
import pandas as pd
import multiprocessing as mp
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale
from matplotlib.backends.backend_pdf import PdfPages
from simiclasso.gene_id_to_name import load_dict, save_dict


def filter_pseudo_BIC(gene_target):
    assert gene_target in tmp.columns.tolist(), f'{gene_target} not present in weight matrix'
    all_tf = tmp[gene_target].abs().sort_values(ascending = False).index.tolist()
    tf_counter = 1
    tf_2_keep = all_tf[:tf_counter]
    while( sum( tmp.loc[tf_2_keep, gene_target]**2 ) / sum(tmp[gene_target]**2) < 0.9):
        tf_counter += 1
        tf_2_keep = all_tf[:tf_counter]
    tf_2_remove = set(all_tf).difference(tf_2_keep)
    tmp.loc[tf_2_remove, gene_target] = 0
    return(tmp[gene_target])

def plot_hist(hist_data, path2plot):
    path2plot = os.path.join(path2plot, 'Target_W.pdf')
    with PdfPages(path2plot) as pdf:
        for data in hist_data:
            plt.figure()
            _ = plt.hist(hist_data[data], bins='auto')  # arguments are passed to np.histogram  
            plt.title(f"Histogram for the number of TFs per target for state {data}")
            pdf.savefig()



def get_stat_degree(matrix):
    binarized = np.where(matrix['weight_dic'][0] !=0 , 1,0)
    degrees = np.sum(binarized, axis =0)
    med  = np.median(degrees)
    maxi = np.max(degrees)
    mini = np.min(degrees)
    return med, maxi, mini

def filter_Ws(path2data, plot_histogram = False, path2plot = None):

    if isinstance(path2data, dict):
        if all(keys in path2data.keys() for keys in ('weight_dic', 'TF_ids', 'query_targets')):
            adata = path2data
    elif os.path.isfile(str(path2data)) and path2data.endswith('.pickle'):
        #if is file and pickle, read the pickle
        print(f'Reading {path2data}')
        adata = pd.read_pickle(path2data)
        # raise OSError(f'{file_name} does not exist')

    plotter = {}
    for w_name in adata['weight_dic'].keys():
        # save the bias data
        bottom_row =  adata['weight_dic'][w_name][-1,]
        # get the weigth data and annotation
        global tmp
        tmp = adata['weight_dic'][w_name][:-1,]
        tmp = pd.DataFrame(scale(tmp, axis =0, with_mean = False, with_std = True), \
        index = adata['TF_ids'], columns = adata['query_targets'])
        # filter the weigths
        pool = mp.Pool(50)
        filtered_weigths = pool.map(filter_pseudo_BIC,  tmp.columns.tolist())
        pool.close()
        # regenerate the correct matrix
        filtered_data = pd.concat(filtered_weigths, axis=1, keys=tmp.columns.tolist())
        # reconvert to numpy and add the bias
        filtered_data = filtered_data.to_numpy()
        plotter[w_name] = (filtered_data != 0).sum(axis=0)
        filtered_data = np.concatenate((filtered_data, bottom_row[None,:]), axis = 0)
        adata['weight_dic'][w_name] = filtered_data

    if plot_histogram:
        plot_hist(plotter, path2plot)

    if os.path.isfile(str(path2data)):
        p2saved_file = os.path.splitext(path2data)
        p2saved_file = p2saved_file[0] + '_Filtered' + p2saved_file[1]
        dict_to_save = {'weight_dic'         :  adata['weight_dic'],
                        'adjusted_r_squared' :  adata['adjusted_r_squared'],
                        'standard_error'     : adata['standard_error'],
                        'TF_ids'             : adata['TF_ids'],
                        'query_targets'      : adata['query_targets']
                        }
        print(f'filtered data saved at: {p2saved_file}')
        save_dict(dict_to_save, p2saved_file)
    else:
        return adata

def filter_and_degree(data):
    weight = {'weight_dic' : data}
    num_TF, num_target = weight['weight_dic'][0].shape
    weight['TF_ids'] = list(map(lambda x: 'TF_' + str(x), range(num_TF-1)))
    weight['query_targets'] = list(map(lambda x: 'target_' + str(x), range(num_target)))
    filtered = filter_Ws(weight)
    binarized = np.where(filtered['weight_dic'][0] !=0 , 1,0)
    median, maxi, mini = get_stat_degree(filtered)
    # print('Degree stats: median {0}, maximum {1}, minimum {2}'.format(median, maxi, mini))
    # print('Degree stats: median {0}'.format(median))
    return median



# print('Yay!')