 
library(reticulate)
reticulate::use_python("/usr/bin/python3")
py_discover_config("magic")
library(Rmagic)


hemato_data_raw <- red.csv('./hemato_data_raw.txt', sep ='\t')

### RUN MAGIC #### #MAGIC works on a cells x genes matrix, seurat gives a genes x cells matrix
hemato_data_raw<-Rmagic::library.size.normalize(t(hemato_data_raw))
hemato_data_raw <- sqrt(hemato_data_raw)
data_MAGIC <- magic(hemato_data_raw,genes='all_genes') 
data_MAGIC_df <- as.data.frame(data_MAGIC)
saveRDS(data_MAGIC_df, './Magic_imputed_data.rds')
