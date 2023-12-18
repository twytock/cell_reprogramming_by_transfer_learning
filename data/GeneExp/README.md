## Notes about input files

Run `code/DataDownloader.py` to obtain the source files to this directory.

Source files include:

1. `GSM_to_CellType_final.csv` -- table annotating each sample to its cell type
2. `ReprogrammingExperiments.xlsx` -- table describing the reprogramming experiments used in validation
3. `KNNGP2NM_ser.csv` -- mapping between the KNN category labels and numbers
4. `all_genexp_data.xlsx` -- table describing the gene expression data
5. `cell_type_phenotype.csv` -- table mapping cell types to phenotypes (similar cell types may be more grouped into broader phenotypes)
6. `delta_corr_newnci60-2.csv.gz` -- table containing the transcriptional response matrix
7. `delta_corr_newnci60.csv.gz` -- table containing the transcriptional response matrix
8. `deltas_reprog_nci60.csv` -- table containing the transcriptional responses observed in reprogramming experiments
9. `fig2_ct_list.txt` -- list of cell types to analyze
10. `fig2_cell_line_data.tar.gz` -- tar archive of source data containing initial states to generate figure 2
11. `fig2_sel_deltas.csv.gz` -- transcriptional responses expressed in the space of gene expression, used for generating figure 2
12. `fig2_sel_pert2gn_ser.csv` -- table mapping columns of the perturbation response matrix to specific genes. Only genes whose expression is measured are included.
13. `gpl570_entrezg_rowmap.txt` -- table mapping the rows of the gene expression matrix to their gene symbols
14. `matched_tissues_for_validation.csv` -- table mapping tissues to specific experiments
15. `non_rpg_delt_columns.txt` -- columns of the transcriptional response matrix that are NOT derived from reprogramming experiments
16. `nonseq_bc_corr_data_all.csv` -- transcriptional data
17. `pert2gn_ser.csv` -- mapping of the perturbations to gene symbols
18. `perturbation_metadata.xlsx` -- table describing the experimental factors of each perturbation
19. `probe_gsym_mapping.csv` -- mapping of microarray probes to gene symbols
20. `reprog_deltas_columns.txt` -- columns of the transcriptional response matrix that ARE derived from reprogramming experiments
21. `tissue_feat_l_WC_all.txt` -- ordered list of eigengenes that distinguish cell types
22. `unpert_ct_inds.csv` -- non-perturbed cell types used for training the KNN model

