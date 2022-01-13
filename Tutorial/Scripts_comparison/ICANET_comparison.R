library(doParallel)
library(Seurat)
library(SingleCellExperiment)
library(cowplot)
library(ica)
library(ICAnet)
library(RcisTarget)
library(mclust)
library(RSCORE)




hepato_data <- readRDS('/home/sevastopol/data/gserranos/SimiC/Simic_Validation/Data/Hepatocytes_seurat_object.rds')

Idents(hepato_data) <- hepato_data$orig.ident
hepato_data <- subset(hepato_data, idents=c('Adult', 'PHx24', 'PHx48', 'PHx96'))
hepato_data$orig.ident <- factor(hepato_data$orig.ident, levels=c('Adult', 'PHx24', 'PHx48', 'PHx96'))
batch <- c(hepato_data$orig.ident)
cluster <- c(hepato_data$State_Functional)
hepato_data$batch <- batch
hepato_data$cluster <- cluster

# integration
hepato_data.list <- SplitObject(hepato_data, split.by='batch')
hepato_data.list <- lapply(X = hepato_data.list, FUN = function(x) {
    x <- NormalizeData(x)
    x <- FindVariableFeatures(x, selection.method = "vst", nfeatures = 2000)
})

features <- SelectIntegrationFeatures(object.list = hepato_data.list, nfeatures=3000)
hepato_data.all <- hepato_data.list[[1]]
hepato_data.all <- GetAssayData(hepato_data.all)[features,]
for (i in 2:length(table(hepato_data$batch))){
    hepato_data.set <- hepato_data.list[[i]]
    hepato_data.set <- GetAssayData(hepato_data.set)[features,]
    hepato_data.all <- cbind(hepato_data.all, hepato_data.set[features,])
}

hepato_data[['Consensus.RNA']] <- CreateAssayObject(hepato_data.all)
DefaultAssay(hepato_data) <- 'Consensus.RNA'
hepato_data <- ScaleData(hepato_data)

hepato_data <- RunPCA(hepato_data, npcs=50, verbose=FALSE, features=rownames(hepato_data))
hepato_data <- RunUMAP(hepato_data, reduction='pca', dims=1:20, reduction.name='umap', reduction.key='umap_')


hs_network_matrix <- getPPI_String(hepato_data, species=10090)
ica.hepato_data <- ICAcomputing(hepato_data, ICA.type = 'JADE', two.stage=FALSE, global.mode=FALSE, center=FALSE, scale=TRUE)
saveRDS(ica.hepato_data, './Data/ica.hepato_data_icaComputing.rds')

pdf('./Plots/heatmap_hepato.pdf')
ica.filter <- CrossBatchGrouping(ica.hepato_data$ica.pooling, cor='spearman', Unique.Preservation=FALSE)
dev.off()


hepato_data <- RunICAnet(hepato_data, ica.filter$ica.filter, PPI.net = hs_network_matrix, W.top=2, aucMaxRank=300, scale=TRUE, cores=48)
hepato_data <- RunModuleSVD(hepato_data, nu=30, power=0.2)
hepato_data <- RunUMAP(hepato_data, reduction = 'Module_SVD', dims=1:20, reduction.name='umap', reduction.key = 'umap_', verbose=FALSE)



hepato_data@active.ident <- as.factor(hepato_data$cluster)
# beta here is the cell type
modules <- FindMarkerModule(hepato_data, identity='Quiescent-state')


pdf('./Plots/test_Umap_Hepatoicascore.pdf')
FeaturePlot(hepato_data, 'ICAnet-1-96-13', reduction='umap')
dev.off()



saveRDS(hepato_data, './Data/ICA_results_hepato.rds')


#### CLUSTERING
DefaultAssay(hepato_data) <- 'Consensus.RNA'

load('./Data/motifAnnotations_mgi_v8.rdata')

Motif_Net <- TF_Net_Generate('./Data/mm9-500bp-upstream-10species.mc8nr.feather', cutoff=1)
Ica.hepato_data <- ICAcomputing(hepato_data, ICA.type = 'JADE', RMT=TRUE, two.stage=FALSE)
hepato_data <- RunICAnetTF(hepato_data, Ica.hepato_data$ica.pooling, W.top.TFs=3, W.top.genes=2.5, aucMaxRank=600, Motif_Net=Motif_Net, TF_motif_annot= motifAnnotations_mgi_v8, cores =48)

moduleInfor <- hepato_data@misc$IcaNet_geneSets_TF_moduleInfor


hepato_data <- RunPCA(hepato_data, npcs =30, features=rownames(hepato_data), verbose=FALSE)
pdf('./Plots/elbow_clusters_hepato.pdf')
ElbowPlot(hepato_data, ndims=30)
dev.off()

hepato_data <- RunTSNE(hepato_data, reduction='pca', dims=1:20, reduction.name='tsne', reduction.key='tSNE_')
pdf('./Plots/tsne_hepato_ica.pdf')
DimPlot(hepato_data, reduction='tsne', group.by='State_Functional', label=1)
DimPlot(hepato_data, reduction='tsne', group.by='orig.ident', label=1)
dev.off()

hepato_data <- FindNeighbors(hepato_data, dims=1:20, reduction='pca')

hepato_data$seurat_clusters <- NULL
hepato_data <- FindClusters(hepato_data, resolution= 0.4, algorithm=2)

pdf('./Plots/tsne_hepato_ica2.pdf')
DimPlot(hepato_data, reduction='tsne', group.by='State_Functional', label=1)
DimPlot(hepato_data, reduction='tsne', group.by='seurat_clusters', label=1)
dev.off()

adjustedRandIndex(as.numeric(as.factor(hepato_data$State_Functional)) , as.numeric(as.factor(hepato_data$seurat_clusters)))


saveRDS(hepato_data ,  './Data/ICA_results_hepato_Finale.rds')