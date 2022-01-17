library(Seurat)

get_hepatodata <- function(path_2_data){
    hepato_data_seurat_raw <- readRDS(path_2_data)
    Idents(hepato_data_seurat_raw) <- hepato_data_seurat_raw$orig.ident
    hepato_data_seurat_raw <- subset(hepato_data_seurat_raw, idents=c('Adult', 'PHx24', 'PHx48', 'PHx96'))
    hepato_data_seurat_raw <- NormalizeData(hepato_data_seurat_raw, normalization.method = "LogNormalize", scale.factor = 10000)
    hepato_data_seurat_raw <- FindVariableFeatures(hepato_data_seurat_raw, selection.method = "vst", nfeatures = 2000)
    hepato_data_seurat_raw <- ScaleData(hepato_data_seurat_raw, assay = 'RNA')
    hepato_data_seurat_raw <- RunPCA(hepato_data_seurat_raw, assay = 'RNA')
    hepato_data_seurat_raw <- RunUMAP(hepato_data_seurat_raw, reduction = "pca", dims = 1:50, min.dist = 0.75)
    hepato_data_seurat_raw <- FindNeighbors(hepato_data_seurat_raw, reduction = "pca", dims = 1:50)
}


ARI_plotter <- data.frame(method=NULL, ARI=NULL, dataset=NULL)
hepato_data_seurat_raw <- get_hepatodata(paste0('/home/sevastopol/data/gserranos/SimiC/Simic_Validation/Data/','Hepatocytes_seurat_object.rds'))
hepato_data_seurat_raw_newClustering <- Seurat::FindClusters(hepato_data_seurat_raw, resolution= 0.1, algorithm=2)
tmp_seurat_raw <- setNames(as.data.frame(hepato_data_seurat_raw_newClustering$seurat_clusters), 'Seurat')
tmp_seurat_raw$cell_id <- rownames(tmp_seurat_raw)
tmp_seurat_raw <- merge(tmp_seurat_raw, hepato_ann, by='cell_id')
ARI_plotter <- rbind(ARI_plotter, data.frame(method='Seurat', ARI=mclust::adjustedRandIndex(as.numeric(as.factor(tmp_seurat_raw$population)) , as.numeric(as.factor(tmp_seurat_raw$Seurat))), dataset='Hepatocytes'))
