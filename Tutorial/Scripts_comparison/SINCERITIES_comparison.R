library(glmnet) 
library(ppcor) 
library(cvTools)

# *** Data loading ***
uploading <- dget("SINCERITIES functions/uploading.R")
# DATA <- uploading(’THP1 data/THP1_single_cell_data_EXCEL_no6_24_72_96.csv’) 
DATA <- uploading('synthetic_d50t20_5000_cells_w_header.csv')

# *** SINCERITIES ***
SINCERITIES_PLUS <- dget("SINCERITIES functions/SINCERITIES_PLUS.R") 
result <- SINCERITIES_PLUS(DATA,noDIAG = 0,SIGN = 1,CV_nfolds = 10) 
adj_matrix <- result$adj_matrix
SIGN <- 1
# Final ranked list
adj_matrix <- adj_matrix/max(adj_matrix)
final_ranked_predictions <- dget("SINCERITIES functions/final_ranked_predictions.R") 
table <- final_ranked_predictions(adj_matrix,DATA$genes,SIGN=1,
		directory = "Results", fileNAME='synthetic_d50t20_simic_cmp',saveFile = TRUE)