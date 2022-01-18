
library(scImpute)
library(Seurat)


# *** Data loading ***
hemato_data <- readRDS('/home/gserranos/dato-activo/03_Analysis/gserranos/NewImputationMethods/granja.rds')

# Analysis of the data: PBMC Vs BMMC
cell_populations <- c('BMMC', 'PBMC')


Idents(hemato_data) <- hemato_data$orig.ident
hemato_data <- subset(hemato_data, idents=cell_populations)

# keep only the labels with more than 10 representants in both groups
# table(hemato_data$labels, hemato_data$orig.ident)
stats <- setNames(as.data.frame(table(hemato_data$labels[hemato_data$orig.ident == 'PBMC']), stringsAsFactors =F), c('label', 'PBMC'))
stats <- merge( stats, setNames(as.data.frame(table(hemato_data$labels[hemato_data$orig.ident == 'BMMC']), stringsAsFactors =F), c('label', 'BMMC')), by= 'label')

hemato_data_raw <- as.data.frame(hemato_data@assays$RNA@counts)
unexpresed_genes <- names(which(rowSums(abs(hemato_data_raw))<1e-6))
hemato_data_raw <- hemato_data_raw[ !rownames(hemato_data_raw) %in% unexpresed_genes, ]

write.table(hemato_data_raw, file='./hemato_data_raw.txt', sep='\t', quote=FALSE)

scimpute('./hemato_data_raw.txt',
		infile='txt',
		outfile='txt',
		out_dir = '/home/gserranos/dato-activo/03_Analysis/gserranos/NewImputationMethods/results/',
		labeled = TRUE,
		drop_thre = 0.5,
		labels = hemato_data$labels,
		ncores = 10)


(sum(hemato_data_raw == 0) /prod(dim(hemato_data_raw)))*100

results <- read.delim('./results/scimpute_count.txt',sep=" ")
(sum(results == 0) /prod(dim(results)))*100
