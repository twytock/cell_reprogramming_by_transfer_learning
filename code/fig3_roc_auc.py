#!/usr/bin/env python
# encoding: utf-8
"""
fig3_roc_auc.py

Created by Thomas Wytock on 2023-11-01.
"""

import time,sys,os
from glob import glob
import os.path as osp, pickle as PP, numpy as np
import matplotlib as mpl, matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
from sklearn.metrics import roc_curve,RocCurveDisplay,auc
from scipy.stats import scoreatpercentile
plt.rcParams['svg.fonttype'] = 'none'

def plot_figS2():
    """Generates Fig. S2, which summarizes the reprogramming vs. non-reprogramming perturbations.
    SVG files are written to the `output/figS2/` directory."""
    pmeta = pd.read_excel('data/GeneExp/perturbation_metadata.xlsx',header=0,index_col=0)
    pmdf = pd.DataFrame([ind.split(';') for ind in pmeta.index if ind.startswith('GSE')])
    VCS2 = pmdf.iloc[:,0].value_counts()    
    xvals = np.arange(VCS2.shape[0])
    yvals = VCS2.values
    xtls = VCS2.index.tolist()
    fig,ax = plt.subplots(1,1,dpi=300,figsize=(1.6,2.25))
    img = ax.bar(xvals,yvals,color='C3',alpha=0.5)
    ax.set_ylabel('Number of reprogramming protocols',size=8,font='Arial')
    ax.set_xlabel('GEO series number',size=8,font='Arial')
    ax.set_xticks(np.arange(len(xtls)))
    ax.set_xticklabels(xtls,size=6,rotation=90,horizontalalignment='center',font='Arial')
    if not osp.exists('output/figS2/'):
        os.makedirs('output/figS2/')
    fig.savefig('output/figS2/figS2_main.svg')
    
    with open('data/GeneExp/reprog_deltas_columns.txt') as fh:
        reprog_deltas = [ln.strip() for ln in fh]
    
    with open('data/GeneExp/non_rpg_delt_columns.txt') as fh:
        non_rpg_delt = [ln.strip() for ln in fh]   
    ## inset
    size_d = {'Reprog.':len(reprog_deltas), 'Other':len(non_rpg_delt)}
    size_ser = pd.Series(size_d)
    fig,ax = plt.subplots(1,1,figsize=(0.5,1.5),dpi=300)
    ax.bar(np.arange(2),size_ser.values,width=0.75,color=['C0','C3'])
    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(size_ser.index.tolist(),rotation=15,horizontalalignment='right',size=6)
    plt.setp(ax.get_yticklabels(),size=6)
    ax.set_ylabel('Number of perturbations',size=6)
    if not osp.exists('output/figS2/'):
        os.makedirs('output/figS2/')
    fig.savefig('output/figS2/figS2_inset.svg')
    return
    

def plot_auc_figure():
    """Generates Fig. 3, which demonstrates the ability of our method to recall reprogramming transitions.

    This function requires the `reprog_validation.py` code to be run.
    An SVG file is written to the `output/fig3/` directory."""
    pmeta = pd.read_excel('data/GeneExp/perturbation_metadata.xlsx',header=0,index_col=0)
    base_fpr = np.linspace(0,1,101)
    fig,ax = plt.subplots(1,1,dpi=300,figsize=(3.375,3))
    tprs_l = []
    roc_auc_d = {}
    for alg in ['A','I','E']:
        if alg=='A':
            clr = 'C3'
        elif alg=='I':
            clr = 'C2'
        elif alg=='E':
            clr = 'C0'
        for fn in glob(f'output/FS_GeneExp/GSE*_{alg}-WC.pkl'):
            tst = pd.read_pickle(fn)
            if not 'pert' in tst.columns:
                print(fn,"corrupted")
                continue
            #tst = pd.read_pickle('GeneExp/Output/FS/GSE12390_nc,foreskin_fibroblast-oe,iPS_A-WC.pkl')
            avg = tst.set_index(['GSMi','GSMf']).groupby('pert').mean().sort_values('pf',ascending=False)
            try:
                TFavg = pmeta.loc[avg.index,'Type']=='RPG'
            except KeyError:
                print(fn,'has nonoverlapping keys')
                continue
            fpr, tpr, thresholds = roc_curve(TFavg, avg.pf)
            esttpr = np.interp(base_fpr,fpr,tpr)
            roc_auc = auc(fpr, tpr)
            tprs_l.append(esttpr)
            lbl = fn.split('/')[-1].split(f'{alg}-WC')[0]
            roc_auc_d[(alg,lbl)]=roc_auc
    
        MU = np.median(np.c_[tprs_l],axis=0)
        AUC = pd.Series(roc_auc_d).unstack(0).median(axis=0).loc[alg]
        l2d = ax.plot(np.r_[0,base_fpr],np.r_[0,MU],color=clr,lw=2,
                      label=f'{alg}: {AUC:.2f}' %(alg,AUC))
    
        UB = scoreatpercentile(np.c_[tprs_l],75,axis=0) 
        LB = scoreatpercentile(np.c_[tprs_l],25,axis=0) 
        ax.fill_between(base_fpr,UB,LB,color=clr,alpha=0.2)
    ax.plot(base_fpr,base_fpr,ls='--',color='k')
    ax.legend(title='AUC',frameon=False,prop={'size':6})
    plt.setp(ax.get_yticklabels(),size=6)
    plt.setp(ax.get_xticklabels(),size=6)
    ax.set_xlabel('False positive rate',size=8)
    ax.set_ylabel('True positive rate',size=8)
    if not osp.exists('output/fig3/'):
        os.makedirs('output/fig3/')
    fig.savefig('output/fig3/fig3_rocauc.svg')
    return

def main():
    plot_figS2()
    plot_auc_figure()

if __name__ == '__main__':
    main()