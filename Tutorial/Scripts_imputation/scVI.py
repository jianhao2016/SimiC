
import scvi
import scanpy as sc

adata = sc.read_text('hemato_data_raw.txt', delimiter = '\t')
sc.pp.filter_genes(adata, min_counts=3)
adata.layers["counts"] = adata.X.copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
adata.raw = adata # freeze the state in `.raw`

sc.pp.highly_variable_genes(
    adata,
    n_top_genes=1200,
    subset=True,
    layer="counts",
    flavor="seurat_v3",
)

scvi.model.SCVI.setup_anndata(
    adata,
    layer="counts"
)

model = scvi.model.SCVI(adata)
model.train()


adata.obsm["X_scVI"] = model.get_latent_representation()
adata.obsm["X_normalized_scVI"] = model.get_normalized_expression()
model.save("my_model/")
model = scvi.model.SCVI.load("my_model/", adata, use_gpu=False)

model.get_normalized_expression().to_csv("imputed_scVI.csv")
adata.obsm["X_normalized_scVI"]