This tutorial is meant to provide a reproducible example of running SimiC and generating some of the images from the manuscript's Figure 5.


#### Input data for SimiC

For the correct run of SimiC, we must provide 3 basic files:
1.  A file containing the single cell normalized gene expression matrix, where the rows are the different cells and the columns are the selected genes, Transcription Factors (TFs) and Targets.
    This input example has been created removing the unexpressed genes, inputing with `Rmagic` the missing values and selecting the top 100 Median Absolute Deviation (MAD) TFs and 1000 MAD Targets.
    This expression matrix should look like this:

```
                          JUN       FOS    NFKBIA      JUNB      LYAR      HOPX     IKZF3  ...   ST3GAL1    ANXA11       PAM  LINC00969       PXN     IL2RB     ITGB1
AAACCTGAGCCCAACC.5   1.114081  0.280526  2.285713  1.091697  1.296435  0.936788  0.843925  ...  0.492110  1.550846  0.461818   0.531912  0.774842  0.540468  0.993670
AAACCTGAGGAGTACC.5   1.472888  0.302284  2.081637  1.490089  1.318613  0.867266  0.832655  ...  0.403069  1.269296  0.350668   0.472743  0.586307  0.504504  0.856413
AAACCTGAGTCTCGGC.5   1.349120  0.276187  2.278155  1.244923  1.409007  1.059473  0.817769  ...  0.497516  1.483029  0.360049   0.471141  0.702477  0.540106  1.003355
AAACCTGCAAGTAGTA.5   0.948583  0.227718  2.247337  0.991819  1.365892  0.469842  0.885265  ...  0.485358  1.483072  0.395588   0.439151  0.523937  0.448124  0.808275
AAACCTGCAATCGGTT.5   1.532367  0.278659  2.074379  1.494496  1.168433  0.910716  0.834177  ...  0.332439  1.134330  0.359519   0.453728  0.618175  0.485554  0.810780
...                       ...       ...       ...       ...       ...       ...       ...  ...       ...       ...       ...        ...       ...       ...       ...
TTGGAACTCGACAGCC.16  2.317779  1.208329  0.633222  1.560164  0.409050  1.487222  0.751372  ...  0.722638  1.240772  0.289974   0.696748  0.038155  0.630867  0.305179
TTGTAGGAGTTAGGTA.16  3.147024  1.871586  0.770267  1.713752  0.095524  0.420563  0.258772  ...  0.641951  0.903628  0.189167   0.779323  0.434905  0.456532  0.438665
TTTACTGAGACCACGA.16  3.200602  2.131516  0.704052  1.842496  0.062911  0.507311  0.196108  ...  0.679146  0.984724  0.163086   0.449529  0.383374  0.529797  0.483403
TTTATGCCAATGAAAC.16  3.853270  2.008095  0.673037  1.771195  0.267704  0.300280  0.228770  ...  0.690994  0.777098  0.131536   0.968869  0.411309  0.311096  0.281270
TTTGGTTGTACAAGTA.16  3.034936  1.899784  0.836446  1.678182  0.055009  0.344758  0.165027  ...  0.554094  0.935291  0.200028   0.690712  0.433391  0.454693  0.484013
```

2.  A file containing the list of the selected TFs. In this case we selected 100 TFs.
3.  A file containing the assignation of the cells to the clusters we want to study. In this case we want to study the regulatory differences driving the CAR-T cells at two different time-points, at the Infusion Point (IP) and at the time the CAR-T cells are more active, de 12th day (D12). This assignation must be numerical, starting at 0, and the order of the assignation is meaningfull. So here we are assigning the IP to the value 0 and D12 to 1. Note that the length of this file must match the number of cells on the expression matrix.


#### Running SimiC cross-validation

The code of SimiC is runned on `python-3.6.5`:

``` python
from simiclasso.clus_regression import simicLASSO_op
from simiclasso.weighted_AUC_mat import main_fn

p2df = 'ClonalKinetics_filtered.DF.pickle' #Path to the gene expression matrix
p2assignment = 'ClonalKinetics_filtered.clustAssign.txt' #Path to the assignment file
p2tf = 'ClonalKinetics_filtered.TFs.pickle' #Path to the list of TFs
p2saved_file = os.path.join('./Results/', 'ClonalKinetics_filtered_CrosVal_Ws.pickle') #Path to the results file
similarity = True #
max_rcd_iter = 100000
df_with_label = False #If the assignment of the cells are provided on a separate file, like in our case, set to FALSE.
cross_val = True #Perform the cross validation to asses the best parameters.
simicLASSO_op(p2df, p2assignment, similarity, p2tf, p2saved_file,  k_cluster, num_TFs, num_target_genes, 
                max_rcd_iter = max_rcd_iter, df_with_label = df_with_label,
                cross_val=cross_val)
```

Once the cross validation of SimiC is complete, we can see the results for the different Lambda1 and Lambda2:

```
	Lambda1	Lambda2	R2
0	0.1	0.1	0.2756
1	0.1	0.01	0.3368
2	0.1	0.001	0.3530
3	0.1	0.0001	0.3406
4	0.1	1e-05	0.3599
5	0.01	0.1	0.8345
6	0.01	0.01	0.8422
7	0.01	0.001	0.8432
8	0.01	0.0001	0.8652
9	0.01	1e-05	0.8672
10	0.001	0.1	0.8296
11	0.001	0.01	0.8545
12	0.001	0.001	0.8622
13	0.001	0.0001	0.8632
14	0.001	1e-05	0.8632
15	0.0001	0.1	0.8448
16	0.0001	0.01	0.8723
17	0.0001	0.001	0.8836
18	0.0001	0.0001	0.8843
19	0.0001	1e-05	0.8845
20	1e-05	0.1	0.8456
21	1e-05	0.01	0.8733
22	1e-05	0.001	0.8856
23	1e-05	0.0001	0.8865
24	1e-05	1e-05	0.8865
```

We decide to set the Lambda1 to 0.01 and the Lambda2 to 0.01.
Now we can run SimiC setting these parameters and perform the weighted AUC calculations.


``` python
from simiclasso.clus_regression import simicLASSO_op
from simiclasso.weighted_AUC_mat import main_fn

p2df = 'ClonalKinetics_filtered.DF.pickle' #Path to the gene expression matrix
p2assignment = 'ClonalKinetics_filtered.clustAssign.txt' #Path to the assignment file
p2tf = 'ClonalKinetics_filtered.TFs.pickle' #Path to the list of TFs
lambda1 = 0.01
lambda2 = 0.01
p2saved_file = './Results/ClonalKinetics_filtered_L1{0}_L2{1}_Ws.pickle'.format(lambda1, lambda2) #Path to the results file
similarity = True #
max_rcd_iter = 100000
df_with_label = False #If the assignment of the cells are provided on a separate file, like in our case, set to FALSE.
simicLASSO_op(p2df, p2assignment, similarity, p2tf, p2saved_file,  k_cluster, num_TFs, num_target_genes, 
                max_rcd_iter = max_rcd_iter, df_with_label = df_with_label,
                lambda1=lambda1, lambda2 = lambda2)

p2AUC = './Results/ClonalKinetics_filtered_L1{0}_L2{1}_AUCs.pickle'.format(lambda1, lambda2)
main_fn(p2df, p2saved_file, p2AUC, percent_of_target = percent_of_target)

```


### Post analysis for the SimiC output

Once we have the final results from SimiC, comprising the weigth file (`*_Ws.pickle`) and the pondered area under the curve file (`*_AUCs.pickle`) we can use the provided script to analyse these results and obtain the images from the paper.

This script is provided in: `Tutorial/SimiC_Post.R`
Additionaly, this script needs the original scRNAseq data, in our case a Seurat object, also provided in: `Tutorial/Data/clonalKinetics_Example.rds`
