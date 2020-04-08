detach('package:reticulate', unload = TRUE)
library(reticulate)
simic <- import('simiclasso')

main_fn <- simic$clus_regression$simicLASSO_op

p2df <- '/data/jianhao/hepatocyte_update_dataset_101619/Bee_DF.pickle'
p2assignment <- '/data/jianhao/hepatocyte_update_dataset_101619/Bees_assignment'
k_cluster <- as.integer(2)
similarity <- TRUE
p2tf <- '/data/jianhao/hepatocyte_update_dataset_101619/Bee_TFs.pickle'
p2saved_file <- '/srv/data/idoia/tmp_test_1119'
num_TFs <- as.integer(5)
num_target_genes <- as.integer(10)

max_rcd_iter <- as.integer(100)
df_with_label <- FALSE

foo<-main_fn(p2df = p2df, p2assignment = p2assignment, k_cluster = k_cluster, similarity = similarity, p2tf = p2tf, 
        p2saved_file = p2saved_file, num_TFs = num_TFs, num_target_genes = num_target_genes, 
        df_with_label = df_with_label, max_rcd_iter = max_rcd_iter)
