#!/usr/bin/env python
# encoding: utf-8
"""
analyze_transitions_final.py

Created by Thomas Wytock on 2023-11-04.
"""

import pickle as P
import numpy as np
import pandas as pd
from glob import glob
import sys,os,os.path as osp
from collections import defaultdict

CH = sys.argv[2]
dmode = sys.argv[1]
OUT = osp.join('output',f'FS_{dmode}',CH)
OUT2 = osp.join('output',dmode) 
IN = osp.join('data',dmode)
K2N_ser = pd.read_csv(osp.join(IN,f'KNNGP2NM_ser.csv'),index_col=0).iloc[:,0]
N2K_ser = K2N_ser.reset_index().set_index('0').iloc[:,0]
    

def analyze_data():
    L = glob(osp.join(OUT,'*-*.pkl'))
    trans_stats_d = defaultdict(dict)
    trans_gnfreq_d = defaultdict(dict)
    for fn in L:
        res_df = pd.read_pickle(fn)
        res_df = res_df.set_index(['GSM_i','CTI','CTF','NG'])
        res_df_stat = res_df.loc[:,'D_F':'KNN_PRB']
        GB = res_df_stat.reset_index(level=3).groupby(level=[0,1,2])
        for (gsmi,cti,ctf),grp in GB:
            trans_stats_d[(gsmi,cti,ctf)]['P_F'] = grp.P_F.max() ## max pct
            trans_stats_d[(gsmi,cti,ctf)]['NG'] = grp.NG.max() ## number of genes
            trans_stats_d[(gsmi,cti,ctf)]['D_F'] = grp.D_F.min() ## closest dist
            path = grp.sort_values('P_F',ascending=True).KNN_GP.values ## path
            num_p = int(N2K_ser.loc[cti])
            cond_path = [cti]
            for ii,num in enumerate(path):
                if num==num_p:
                    continue
                else:
                    cond_path.append(K2N_ser.loc[num])
            trans_stats_d[(gsmi,cti,ctf)]['path'] = tuple(cond_path)
        ##     frequency of presence of particular genes
        ##     the sum of all weights for an experiment is normalized to 1
        ##     the weights of a gene ranked $r$ of $N$ total is : \sum_{k=r}^N 1/k
        res_df_alpha = res_df.iloc[:,4:]
        nz_alpha = (res_df_alpha>0).astype(int).sum(axis=0)
        res_df_alpha = res_df_alpha.T[nz_alpha>0].T
        GB2 = res_df_alpha.reset_index(level=3).groupby(level=[0,1,2])
        for (gsmi,cti,ctf),grp2 in GB2:
            grp2 = grp2.set_index('NG').sort_index()
            srt_inds = grp2.index.tolist()
            for p_tup,col in grp2.items():
                if any(col>0):
                    minval = col[col>0].index.min()
                    IND = srt_inds.index(minval)
                    trans_gnfreq_d[(gsmi,cti,ctf)][p_tup]=sum([1/ii for ii in srt_inds[IND:]])/col.shape[0]
                else:
                    trans_gnfreq_d[(gsmi,cti,ctf)][p_tup]=0
    opdf1 = pd.DataFrame(dict(trans_stats_d))
    opdf1.to_pickle(osp.join(OUT2,f'{CH}_stats_df.pkl'))
    opdf2 = pd.DataFrame(dict(trans_gnfreq_d))
    opdf2.to_pickle(osp.join(OUT2,f'{CH}_gnfreq_df.pkl'))
    return opdf1,opdf2

def gnfreq_analysis():
    df = pd.read_pickle(osp.join(OUT2,f'{CH}_gnfreq_df.pkl')).T
    wt_gn_df = df.groupby(level=[1,2]).mean()
    wt_gn_df.to_pickle(osp.join(OUT2,f'{CH}_wtmean_gnfreq_df.pkl'))
    unwt_gn_df = (df>0).astype(int).groupby(level=[1,2]).mean()
    unwt_gn_df.to_pickle(osp.join(OUT2,f'{CH}_unwtmean_gnfreq_df.pkl'))
    return 0

def stats_analysis():
    df = pd.read_pickle(osp.join(OUT2,f'{CH}_stats_df.pkl')).T
    stats_df = df.loc[:,['P_F','NG','D_F']].groupby(level=[1,2]).mean()
    stats_df.to_pickle(osp.join(OUT2,f'{CH}_mean_stats_df.pkl'))
    return 0

def main():
    trans_stats_df, trans_gnfreq_df = analyze_data()
    gnfreq_analysis()
    stats_analysis()

if __name__ == '__main__':
    main()