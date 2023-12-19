# Instructions for operating the codes
This README file provides instructions for running the codes necessary for reproducing the results of the paper.
All scripts in the code directory must be invoked from the top-level directory of this repository so that
the paths to the `data/` and `output/` directories will be correct.
The scripts rely upon the system requirements mentioned in the README file located at the top-level directory of this respository.

# Obtaining data
## `DataDownloader.py`

This script downloads the source data necessary for obtaining the results and saves it to `data/RNASeq` and `data/GeneExp`.
Data may be downloaded using the command

`python code/DataDownloader.py`

# Comparison with annotation-based methods
## `naive_v_opt_final.py`

This script produces the panels of Fig. 2 of the main text and Fig. S1 of the SI.
It may be called using

`python -m scoop -n 8 code/naive_v_opt_final.py`

where 8 may be changed to the desired number of processors 
to parallelize the computation over.
The script will save the optimal single-gene and all-gene perturbation
prescriptions to the `output/naive_v_opt/` directory and generate the 
panels of Fig. 2 and Fig. S1 in the `output/fig2/` directory.

# Validation against previous reprogramming experiments

## `reprog_vaildation.sh`

This script runs `reprog_validation_final.py` for each of the constraint conditions, \{`A`, `I`, `E`\}, on the control inputs.
It may be called using 

`./code/reprog_vaildation.sh`

Files are saved in `output/FS_GeneExp/`.

## `fig3_roc_auc.py`

This script produces Fig. 3 of the main text and Fig. S2 of the SI.
It may be called using

`python code/fig3_roc_auc.py`

The figures will be generated in the directories `output/figS2` and `output/fig3`.

# Computing and analyzing transdifferentiation transitions
## `fs.sh`

This shell script runs the file `forward_selection_final.py`,
which computes the optimal gene perturbations for each case using Mixed-Integer Quadratic Programming (MIQP).
The shell script takes two positional arguments. 

1. Dataset to operate on, valid values: \{ `RNASeq`, `GeneExp` \}.
2. Constraints on the control inputs $u_\ell$, valid values: \{`A`, `I`, `E`\}. These values correspond to the constraints $-\infty < u_{\ell} < \infty$, $| u_{\ell} |\leq1$, and $0\leq u_{ell} \leq 1$, respectively.

The script may be invoked, for example:

`./code/fs.sh RNASeq E`

To calculate the optimal perturbations subject to the constraints $0\leq u_ell \leq 1$ for all 36 cell types in the RNASeq dataset.
It is recommended that this script be run on a cluster to take advantage of the parallelization, since the runtime to calculate solutions subject to the most restrictive constraint condition for the RNASeq dataset is about 5 days on a cluster of 144 cores.

While running, the script will create temporary directories in the `output/FS_RNASeq` and `output/FS_GeneExp` directories. In the example command above, the files of the form `{cti}-{ctf}.pkl`, where `{cti}` and `{ctf}` are substitution strings for the initial and target cell type pairs, would be stored in the directory `output/FS_RNASeq/E/`.

### `forward_selection_final.py`
This python script may also be invoked for individual cell types, for example:

`python -m scoop -n 8 code/forward_selection_final.py Stomach RNASeq E`

The command line arguments are the target cell type, the dataset, and the constraint condition, in that order.

## `analyze_transitions_final.py`

This script gathers the results created by `fs.sh` and gathers them into single files. 
It uses the same command line arguments as `fs.sh`.
Continuing the example above, the script may be invoked using:

`python code/analyze_transitions_final.py RNASeq E`

The files `output/FS_RNASeq/E/*-*.pkl` would be agglomerated into two files:

1. `output/RNASeq/E_stats_df.pkl`, which contains a table of summary statistics of the transitions like the number of genes needed to transdifferentiate cells, and the fraction of the distance recovered.
2. `output/RNASeq/E_gnfreq_df.pkl`, which contains a table of the control inputs used in each transdifferentiation instance.

These two files are further processed to take means over all cell type pairs, resulting in the files:

1. `output/RNASeq/E_wtmean_gnfreq_df.pkl`, which give a "weighted" average over cell types. The average is weighted by the size of the control inputs of a gene.
2. `output/RNASeq/E_unwtmean_gnfreq_df.pkl`, which give a "unweighted" average over cell types. Each instance of a gene appearing is treated as a 1.
3. `output/RNASeq/E_mean_stats_df.pkl`, which contains the average transition statistics for each cell type pair.


## `make_network_figure_final.py`

This script produces the panels of Fig. 4 of the main text. It also produces the graphml files that 
may be used to create Figs. 5 and 6 of the main text and Figs. S3 and S4 of the SI.
In addition, this script produces Fig. 7.

The script uses the output of `analyze_transitions_final.py` to create the figures. It has a single command line argument, the dataset:

`python -m scoop -n 8 make_network_figure.py RNASeq`

The computation of the exact $p$-values is expensive, so it is recommended to parallelize that computation.

The graphml files are output to the `output/Graphs` directory

The figures are output to directories of the form `output/fig{N}`, where `{N}` is the number of the figure in the paper to which the panels belong.

