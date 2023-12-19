import os
import os.path as osp
import tarfile
import gdown

GeneExp_url_dict = {
    'GSM_to_CellType_final.csv':'https://drive.google.com/uc?id=16NJyRJ8ECp6ddSfbJKF7T1Q61pUxXL4o',
    'KNNGP2NM_ser.csv':'https://drive.google.com/uc?id=1N9iCloIUXPohEaIV9uDRBGCvTL66h-zs',
    'cell_type_phenotype.csv':'https://drive.google.com/uc?id=1DO5r_cw5GmskWP_LJpY2WzS-YJM35evC',
    'delta_corr_newnci60-2.csv.gz':'https://drive.google.com/uc?id=1CObJRy2jVTZ8jHhGTOXZSMk0fyDRR_A6',
    'delta_corr_newnci60.csv.gz':'https://drive.google.com/uc?id=1j567jwnvU42mmdNPC8Vn_M_1SKD4EJWM',
    'deltas_reprog_nci60.csv':'https://drive.google.com/uc?id=1gscOhrn-BeQ9WdYTyPmuwEwwGofJFh2J',
    'fig2_ct_list.txt':'https://drive.google.com/uc?id=1m5C9e39Bn3L0eLVgJm4VOAd96uCigYKc',
    "fig2_cell_line_data.tar.gz":'https://drive.google.com/uc?id=11XbytzXwDMX0Xs8NcD4FO4Iju2BljYAP',    
    "fig2_sel_deltas.csv.gz":'https://drive.google.com/uc?id=1vDCOoVbhIIxYbjtkbUJbX6dEkxuLGQc2',
    "fig2_sel_pert2gn_ser.csv":'https://drive.google.com/uc?id=1Xm8ZvIKUjxY1YtYmLhH5IM7MqPYP0UrZ',
    "gpl570_entrezg_rowmap.txt":'https://drive.google.com/uc?id=1Dw0jzuM_8Smb1PAuX8egrueMOGUK4E-w',
    "matched_tissues_for_validation.csv":'https://drive.google.com/uc?id=1YlF8xSCxjfSfnBC6QmwZx-79daIgus6Q',
    "non_rpg_delt_columns.txt":'https://drive.google.com/uc?id=11JZFvNQkSLYlf3cLsIGQmeSnaRDHTgQf',
    "nonseq_bc_corr_data_all.csv":'https://drive.google.com/uc?id=1EpH64xySRxhVbh_qE8GxkfKPV6zfIoyH',
    "pert2gn_ser.csv":'https://drive.google.com/uc?id=1tUludA9zMBGC-Tep1spAGwG6QH1L3uep',
    "probe_gsym_mapping.csv":'https://drive.google.com/uc?id=1uBr5ZaxzR8C74j4ZzWbnFlfnZDDCbCIl',
    "reprog_deltas_columns.txt":'https://drive.google.com/uc?id=1_foOg7Uy8tmwB28esTpjXEwKJampXNXx',
    "tissue_feat_l_WC_all.txt":'https://drive.google.com/uc?id=1XSRD6-AwgvDCWYfsgjNHFRe2o5phh8s1',
    "unpert_ct_inds.csv":'https://drive.google.com/uc?id=1PxBZmyoCL896Jo05JMegYKtn16kzZlUS',
}

RNASeq_url_dict = {
    "GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt":"https://drive.google.com/uc?id=1v2fEFsViUmrf-e0lHtm6-LqlCuZ7TrBy",
    "GTEx_feat_l_bc.txt":"https://drive.google.com/uc?id=11S6D3DgtPfMUZXySDj1y5ibjbwIXGBJb",
    "KNNGP2NM_ser.csv":"https://drive.google.com/uc?id=1dm6eePbPUNz9MJ96AQrZ1o-Tn2cN1hme",
    "allcts.txt":"https://drive.google.com/uc?id=1EiblCf-LdieGdutEiJt1oXzNEC4Q5444",
    "celltype_color_tissue.csv":"https://drive.google.com/uc?id=1vnUy5Egi_Llw2UtFpu4uxkMU_nBvRxJR",
    "gene_symbol_mappings.txt.gz":"https://drive.google.com/uc?id=16RXUt6l5eJ38t5XN3I-mxNH2lvQEnwss",
    "proj_perturbations_df.csv":"https://drive.google.com/uc?id=1JkQHmUbIHV2op8wQLHR4EbAqYdHrqZzw",
    "proj_data_df.csv":"https://drive.google.com/uc?id=1wsl8Z9L7fGz4fm1hdJkygPSi1iS-WGvA"
}


def download_data(data_name):
    data_path = osp.join("data",data_name)
    if data_name=='GeneExp':
        url_dict = GeneExp_url_dict
    elif data_name=='RNASeq':
        url_dict = RNASeq_url_dict
    for fn,file_url in url_dict.items():
        file_path = osp.join(data_path,fn)
        if not os.path.exists(file_path):
            gdown.download(file_url, file_path, quiet=False)
            if file_path.endswith('tar.gz'):                
                with tarfile.open(file_path,'r:gz') as tfh:
                    tfh.extractall()
                os.remove(file_path)
            

def main():
    data_names = ["RNASeq","GeneExp"]
    for data_name in data_names:
        download_data(data_name)


if __name__ == '__main__':
    main()
