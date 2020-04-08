detach('package:reticulate', unload = TRUE)
library(reticulate)
simic <- import('simiclasso')

main_fn <- simic$clus_regression$simicLASSO_op

p2df <- '../test_data/test_data.pickle'
p2assignment <- '../test_data/test_assignment.txt'
k_cluster <- as.integer(3)
similarity <- TRUE
p2tf <- '../test_data/mouse_TF.pickle'
p2saved_file <- '../test_incident_matrices'
num_TFs <- as.integer(50)
num_target_genes <- as.integer(100)

max_rcd_iter <- as.integer(50000)
df_with_label <- FALSE

foo<-main_fn(p2df = p2df, p2assignment = p2assignment, k_cluster = k_cluster, similarity = similarity, p2tf = p2tf, 
        p2saved_file = p2saved_file, num_TFs = num_TFs, num_target_genes = num_target_genes, 
        df_with_label = df_with_label, max_rcd_iter = max_rcd_iter)


wAUC <- simic$weighted_AUC_mat$main_fn

p2AUC <- '../test_wAUC_matrices'
wAUC(p2df = p2df, p2saved_file = p2saved_file, p2AUC = p2AUC)