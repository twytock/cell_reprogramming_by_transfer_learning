import os
import os.path as osp
import tarfile
import urllib.request

GeneExp_url_dict = {
    "GSM_to_CellType_final.csv":"https://drive.google.com/file/d/16NJyRJ8ECp6ddSfbJKF7T1Q61pUxXL4o/view?usp=share_link",
    "KNNGP2NM_ser.csv":"https://drive.google.com/file/d/1N9iCloIUXPohEaIV9uDRBGCvTL66h-zs/view?usp=share_link",
    "ReprogrammingExperiments.xlsx":"https://docs.google.com/spreadsheets/d/1huDdLKPz6NZycf1LmnKlmznx0q5hj1ba/edit?usp=share_link&ouid=103178190040155494824&rtpof=true&sd=true",
    "all_genexp_data.xlsx":"https://docs.google.com/spreadsheets/d/1hoxZYx4UeO5BTNB4c7pcCSTCcEwzhVEH/edit?usp=share_link&ouid=103178190040155494824&rtpof=true&sd=true",
    "cell_type_phenotype.csv":"https://drive.google.com/file/d/1DO5r_cw5GmskWP_LJpY2WzS-YJM35evC/view?usp=share_link",
    "delta_corr_newnci60-2.csv.gz":"https://drive.google.com/file/d/1CObJRy2jVTZ8jHhGTOXZSMk0fyDRR_A6/view?usp=share_link",
    "delta_corr_newnci60.csv.gz":"https://drive.google.com/file/d/1j567jwnvU42mmdNPC8Vn_M_1SKD4EJWM/view?usp=share_link",
    "deltas_reprog_nci60.csv":"https://drive.google.com/file/d/1gscOhrn-BeQ9WdYTyPmuwEwwGofJFh2J/view?usp=share_link",
    "fig2_ct_list.txt":"https://drive.google.com/file/d/1m5C9e39Bn3L0eLVgJm4VOAd96uCigYKc/view?usp=share_link",
    "fig2_cell_line_data.tar.gz":"https://drive.google.com/file/d/11XbytzXwDMX0Xs8NcD4FO4Iju2BljYAP/view?usp=share_link",
    "fig2_sel_deltas.csv.gz":"https://drive.google.com/file/d/1vDCOoVbhIIxYbjtkbUJbX6dEkxuLGQc2/view?usp=share_link",
    "fig2_sel_pert2gn_ser.csv":"https://drive.google.com/file/d/1Xm8ZvIKUjxY1YtYmLhH5IM7MqPYP0UrZ/view?usp=share_link",
    "gpl570_entrezg_rowmap.txt":"https://drive.google.com/file/d/1Dw0jzuM_8Smb1PAuX8egrueMOGUK4E-w/view?usp=share_link",
    "matched_tissues_for_validation.csv":"https://drive.google.com/file/d/1YlF8xSCxjfSfnBC6QmwZx-79daIgus6Q/view?usp=share_link",
    "non_rpg_delt_columns.txt":"https://drive.google.com/file/d/11JZFvNQkSLYlf3cLsIGQmeSnaRDHTgQf/view?usp=share_link",
    "nonseq_bc_corr_data_all.csv":"https://drive.google.com/file/d/1EpH64xySRxhVbh_qE8GxkfKPV6zfIoyH/view?usp=share_link",
    "pert2gn_ser.csv":"https://drive.google.com/file/d/1tUludA9zMBGC-Tep1spAGwG6QH1L3uep/view?usp=share_link",
    "perturbation_metadata.xlsx":"https://docs.google.com/spreadsheets/d/1xLxxOTNy75qSYMMXXYALTw6-penQO8fH/edit?usp=share_link&ouid=103178190040155494824&rtpof=true&sd=true",
    "probe_gsym_mapping.csv":"https://drive.google.com/file/d/1uBr5ZaxzR8C74j4ZzWbnFlfnZDDCbCIl/view?usp=share_link",
    "reprog_deltas_columns.txt":"https://drive.google.com/file/d/1_foOg7Uy8tmwB28esTpjXEwKJampXNXx/view?usp=share_link",
    "tissue_feat_l_WC_all.txt":"https://drive.google.com/file/d/1XSRD6-AwgvDCWYfsgjNHFRe2o5phh8s1/view?usp=share_link",
    "unpert_ct_inds.csv":"https://drive.google.com/file/d/1PxBZmyoCL896Jo05JMegYKtn16kzZlUS/view?usp=share_link"
}

RNASeq_url_dict = {
    "GTEx_Analysis_v8_Annotations_SampleAttributesDS.txt":"https://drive.google.com/file/d/1v2fEFsViUmrf-e0lHtm6-LqlCuZ7TrBy/view?usp=share_link",
    "GTEx_feat_l_bc.txt":"https://drive.google.com/file/d/11S6D3DgtPfMUZXySDj1y5ibjbwIXGBJb/view?usp=share_link",
    "KNNGP2NM_ser.csv":"https://drive.google.com/file/d/1dm6eePbPUNz9MJ96AQrZ1o-Tn2cN1hme/view?usp=share_link",
    "allcts.txt":"https://drive.google.com/file/d/1EiblCf-LdieGdutEiJt1oXzNEC4Q5444/view?usp=share_link",
    "celltype_color_tissue.csv":"https://drive.google.com/file/d/1vnUy5Egi_Llw2UtFpu4uxkMU_nBvRxJR/view?usp=share_link",
    "gene_symbol_mappings.txt.gz":"https://drive.google.com/file/d/16RXUt6l5eJ38t5XN3I-mxNH2lvQEnwss/view?usp=share_link",
    "proj_perturbations_df.csv":"https://drive.google.com/file/d/1JkQHmUbIHV2op8wQLHR4EbAqYdHrqZzw/view?usp=share_link",
    "proj_data_df.csv":"https://drive.google.com/file/d/1wsl8Z9L7fGz4fm1hdJkygPSi1iS-WGvA/view?usp=share_link"
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
            urllib.request.urlretrieve(file_url,file_path)
            print(f"{fn} data has been downloaded and saved in {file_path}")
            if file_path.endswith('tar.gz'):
                with tarfile.open(file_path,'r:gz') as tfh:
                    head,tail = osp.split(file_path)
                    tpath = tail.split('.tar.gz')[0]
                    tfh.extractall(path=osp.join(head,tpath))
            

def main():
    data_names = ["RNASeq","GeneExp"]
    for data_name in data_names:
        download_data(data_name)


if __name__ == '__main__':
    main()
