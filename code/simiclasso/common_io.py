import pandas as pd
import numpy as np
import pickle
import time
from scipy.sparse import coo_matrix

def preprocessing_expression_mat(X_raw):
    '''
    Perform log2 and Z-transform on raw input expression matrix
    '''
    # X = np.log2(X_raw + 1)
    # columns standarization
    # X = (X - np.mean(X, axis = 0))/(np.std(X, axis = 0) + 1e-5)
    X = X_raw
    return X

def extract_df_columns(df, feature_cols):
    '''
    find match in df, and return extracted numpy array
    '''
    lower_case_fc = [x.lower() for x in feature_cols]
    # match_idx = df.columns.isin(feature_cols)
    match_idx = df.columns.str.lower().isin(lower_case_fc)
    extracted_df = df.loc[:, match_idx]
    # new_feat_cols = extracted_df.columns.values.tolist()
    return extracted_df


def load_dataFrame(df_file, feature_cols):
    '''
    This function takes a pandas dataFrame from disk and turns it into expression matrix 
    and labels
    return:
        X: expression matrix, n x m
            n: number of cells,
            m: number of genes [len of feature columns]
        Y: labels of cells, default 3 types.
    '''
    df = pd.read_pickle(df_file)
    # with open(feature_file, 'rb') as f:
    #     feature_cols = pickle.load(f)
    # X_raw = df[feature_cols].values
    # match_idx = df.columns.isin(feature_cols)
    # X_raw_df = df.loc[:, match_idx]
    X_raw_df = extract_df_columns(df, feature_cols)

    new_feature_cols = X_raw_df.columns.values.tolist()
    X_raw = X_raw_df.values

    X = preprocessing_expression_mat(X_raw)
    Y = df['label'].values

    print('Done loading, shape of X = ', X.shape)
    return X, Y, new_feature_cols


def load_gene_interaction_net(p2f, p2feat_cols, feature_cols):
    '''
    load the gene interaction network, sample the overlapping value between feature_cols
    convert to a networkx class.
    input:
        p2f: path to gene gene interaction network
        p2feat_col: path to feature column file of ggn
        feature_cols: feature columns [list of string] in expression matrix X
    output:
        a adjacency matrix, GGN
        feature/genes that are both in the network and in input feature_cols
    -------
    to construct a sparse matrix, we need:
        C: values of [x, y]
        A: coordinate of x
        B: coordinate of y
    mat = sparse.coo_matrix((C, (A, B)), shape = (n, n))
    '''
    # A, feat_cols = random_adjacency_matrix(feature_cols)

    ggn_df = pd.read_csv(p2f, compression='gzip', error_bad_lines=False)
    ggn_genes = pd.read_csv(p2feat_cols)
    print('GGN includes {} genes'.format(len(ggn_genes)))

    edge_list = ggn_df[['protein1', 'protein2']].values
    weight_list = ggn_df['combined_score'].values

    # node to index dictionary. Contains every genes in both experssion matrix 
    # and in GGN network. values of each gene key will be its index on output list.
    nodes_idx_dict = {}
    output_gene_list = []
    idx = 0  # start index of gene list. Add one each time get a new gene.
    x_coord_list = []
    y_coord_list = []
    mat_val_list = []

    gene_set_in_X = set(feature_cols)

    for edge, weight in zip(edge_list, weight_list):
        node_1, node_2 = edge
        # every nodes should be in the gene list of expression matrix
        if (node_1 in gene_set_in_X) & (node_2 in gene_set_in_X):
            # both genes are included, add val of adjacency.
            mat_val_list.append(weight)

            # add x coordinate
            if node_1 in nodes_idx_dict:
                tmp_x = nodes_idx_dict[node_1]
            else:
                # node_1 is not in dictionary.
                # new gene, add to dict, update idx, and output list
                nodes_idx_dict[node_1] = idx
                tmp_x = idx
                idx += 1
                output_gene_list.append(node_1)
            x_coord_list.append(tmp_x)

            # add y coordinate
            if node_2 in nodes_idx_dict:
                # gene already in dict
                # no need to update output list.
                tmp_y = nodes_idx_dict[node_2]
            else:
                # same argument
                nodes_idx_dict[node_2] = idx
                tmp_y = idx
                idx += 1
                output_gene_list.append(node_2)
            y_coord_list.append(tmp_y)

            assert len(nodes_idx_dict) == idx

    ggn_sparse = coo_matrix((mat_val_list,(x_coord_list,y_coord_list)),
            shape=(idx,idx))
    ggn_sparse += ggn_sparse.T
    # convert back to dense adjacency matrix
    GGN_dense = ggn_sparse.toarray()
    print('GGN generated, size of matrix = {}'.format((idx, idx)))
    
    return GGN_dense, output_gene_list

def split_df_and_assignment(df_in, assignment, test_proportion = 0.2):
    num_of_cells = len(assignment)
    size_of_test_set = int(num_of_cells * test_proportion)
    random_perm = np.random.permutation(num_of_cells)
    test_idx = random_perm[:size_of_test_set]
    train_idx = random_perm[size_of_test_set:]

    train_df = df_in.loc[train_idx]
    train_assign = assignment[train_idx]

    test_df = df_in.loc[test_idx]
    test_assign = assignment[test_idx]
    
    return train_df, test_df, train_assign, test_assign
    


# below are basically useless
def change_expression_matrix_order(X_df, expect_order):
    '''
    change columns in X so that its order is same as expect_order
    return new order X.
    input:
        X: origin expression matrix
        gene_order_in_X: gene order in original matrix
        expect_order: expect gene order after change
    output:
        X_new: same dimension as X, has gene in order expect_order
    '''
    # perm = get_str_permutation(gene_order_in_X, expect_order)
    X_new = X_df[expect_order].values
    feat_cols_new = expect_order
    return X_new, feat_cols_new
    

def get_gene_list_order_in_int(gene_order):
    '''
    convert a string gene list to range(len(gene list)).
    and provide a dictionary, in 
    gene_2_int_dict = {'gene1':index}
    for fast search
    '''
    len_of_gene = len(gene_order)
    gene_order_int = np.arange(len_of_gene)
    gene_2_int_dict = {gene_name:idx for gene_name, idx in zip(gene_order, gene_order_int)}
    return gene_order_int, gene_2_int_dict


def get_int_order_wrt_dict(gene_list, gene_2_int_dict):
    '''
    given some gene list in [string], and 
    a dictionary of form {'gene name': idx}
    convert the list of gene to list of integer
    '''
    gene_list_len = len(gene_list)
    new_list = np.zeros(gene_list_len)
    pass

