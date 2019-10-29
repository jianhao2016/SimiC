# simicLASSO

## Installation:
Please make sure that you are using Python 3.x, and the packages in `requirements.txt` is properly installed. 

## Running the code
To run simicLASSO with Single-cell RNA-seq of mouse cerebral cortex dataset[GSE60361](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE60361)

go to folder `code`, in command line, type in 
```
python clus_regression_mouse.py --similarity 1 --p2df <path to dataframe file> --p2fc <path to feature column file>  --num_target_genes 100
```
This will run simicLASSO with output GRNs of top 100 expressed target genes.
