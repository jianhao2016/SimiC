
# conda activate arboreto-scenic

import pandas as pd

from dask.diagnostics import ProgressBar
from arboreto.utils import load_tf_names
from arboreto.algo import grnboost2
from pyscenic.rnkdb import FeatherRankingDatabase as RankingDatabase
from pyscenic.utils import modules_from_adjacencies, load_motifs
from pyscenic.prune import prune2df, df2regulons
from pyscenic.aucell import aucell
from pyscenic.binarization import binarize
from pyscenic.cli.utils import save_enriched_motifs
import scanpy as sc
import numpy as np
import subprocess as sp

data_path = '/home/sevastopol/data/gserranos/SCENIC/Data/Hepato4Scenic.tsv'
DATA = 'Hepato2020_scenic_'


adata = sc.read_csv(data_path, delimiter = '\t', first_column_names=True).transpose()


adata.var_names_make_unique()
# compute the number of genes per cell (computes ‘n_genes' column)
sc.pp.filter_cells(adata, min_genes=0)
# mito and genes/counts cuts
mito_genes = adata.var_names.str.startswith('MT-')
# for each cell compute fraction of counts in mito genes vs. all genes
adata.obs['percent_mito'] = np.ravel(np.sum(
    adata[:, mito_genes].X, axis=1)) / np.ravel(np.sum(adata.X,
    axis=1))
# add the total counts per cell as observations-annotation to adata
adata.obs['n_counts'] = np.ravel(adata.X.sum(axis=1))

sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
adata = adata[adata.obs['n_genes'] < 4000, :]
adata = adata[adata.obs['percent_mito'] < 0.15, :]


row_attrs = {
"Gene": np.array(adata.var_names),
}
col_attrs = {
"CellID": np.array(adata.obs_names),
"nGene": np.array(np.sum(adata.X.transpose()>0, axis=0)).flatten(),
"nUMI": np.array(np.sum(adata.X.transpose(),axis=0)).flatten(),
}

# load the TF for mouse
tf_names = pd.read_pickle('/home/sevastopol/data/gserranos/SimiC/Simic_Validation/Data/mouse_TF.pickle')
tf_names2 = list(map(lambda x: x.capitalize(), tf_names))


expData = adata.X.toarray()
geneNames = np.array(adata.var_names.tolist(), dtype='object')
sample_names = adata.obs_names.tolist()

adjacencies = grnboost2(expData,gene_names=geneNames, tf_names=tf_names, verbose=True)
adj_file = DATA + 'adj.tsv'
adjacencies.to_csv(adj_file, index=False, sep='\t')

expr_file = DATA + 'expr.tsv'
expData_pd  =  pd.DataFrame(data=expData, index=sample_names, columns=geneNames.tolist())
expData_pd.to_csv(expr_file, index=True, sep='\t')


CMD = 'pyscenic ctx \
Hepato2020_scenic_adj.tsv \
./Data/mm10__refseq-r80__10kb_up_and_down_tss.mc9nr.feather \
--annotations_fname ./Data/motifs-v9-nr.mgi-m0.001-o0.0.tbl \
--expression_mtx_fname  Hepato2020_scenic_expr.tsv \
--mode "dask_multiprocessing" \
--output Hepato2020_scenic_reg.csv \
--num_workers 40 \
--mask_dropouts'

sp.check_call(CMD, shell=True)

CMD = 'pyscenic aucell \
Hepato2020_scenic_expr.tsv \
Hepato2020_scenic_reg.csv \
--output Hepato2020_scenic_auc.tsv \
--num_workers 20'

sp.check_call(CMD, shell=True)

# If the data need the binarization step:
data = pd.read_csv('Hepato2020_scenic_auc.tsv', sep='\t', index_col=0)
bin_data = binarize(data)
bin_data[0].to_csv('Hepato2020_scenic_auc_bin.tsv')

