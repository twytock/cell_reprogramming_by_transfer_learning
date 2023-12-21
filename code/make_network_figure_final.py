#!/usr/bin/env python
# encoding: utf-8
"""
make_network_figure_final.py

Created by Thomas Wytock on 2023-11-03.
"""

import pickle as P
import numpy as np
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['svg.fonttype'] = 'none'
from scipy.stats import binom
from statsmodels.stats.multitest import multipletests
from functools import partial
from scoop import futures
import sys,os,os.path as osp
from collections import defaultdict
from itertools import combinations as COMBS

dmode = sys.argv[1]
OUT = 'output'
IN = 'data'
def annotate_network(stats_df,nn,ff):
    '''
    ARGUMENTS (type) DESCRIPTION
    stats_df (pandas DataFrame) 
        columns: D_F -- final distance 
                 P_F -- % recovered
                 KNN_GP -- cell type
                 NG -- number of genes
        index levels:
                 0 -- GSM_i -- initial state
                 1 -- CTI   -- initial cell type
                 2 -- CTF   -- final cell type
    nn (int) maximum number of genes
    ff (float64) frac of reprogrammed states to call a transition'''
    cti_ctf_frac = {}
    cti_ctf_path = {}
    for (cti,ctf),grp in stats_df.groupby(level=[1,2]):
        cti_ctf_frac[(cti,ctf)] = (grp.NG<=nn).sum()/grp.shape[0]
        if (grp.NG<=nn).sum() >0:
            cti_ctf_path[(cti,ctf)] = grp[grp.NG<=nn].path.value_counts().index[-1]
        else:
            cti_ctf_path[(cti,ctf)] = []
    cti_ctf_ser = pd.Series(cti_ctf_frac)
    cti_ctf_path = pd.Series(cti_ctf_path)
    SEL = cti_ctf_ser[cti_ctf_ser>ff]
    SEL_path = cti_ctf_path[cti_ctf_ser>ff]
    G = nx.DiGraph()
    G.add_edges_from(SEL.index.tolist())
    for node in G.nodes():
        G.nodes[node]['name']=str(node)
    for u,v in G.edges():
        G.edges[u,v]['weight'] = float(SEL.loc[(u,v)])
    H = nx.DiGraph()
    H_edges = []
    edge_def_d = defaultdict(int)
    for (cti,ctf),spath in SEL_path.items():
        if not cti==spath[0]:
            spath = [cti]+list(spath)
        if not ctf==spath[-1]:
            spath = list(spath)+[ctf]
        for u,v in zip(spath[:-1],spath[1:]):
            edge_def_d[(u,v)]+=1
    edge_def_ser = pd.Series(dict(edge_def_d))
    H.add_edges_from(edge_def_ser.index.tolist())
    for node in H.nodes():
        H.nodes[node]['name']=str(node.replace(' - ','\n'))
    for u,v in H.edges():
        H.edges[u,v]['weight']=int(edge_def_ser.loc[(u,v)])
    if not osp.exists(osp.join(OUT,'Graphs')):
        os.makedirs(osp.join(OUT,'Graphs'))
    nx.readwrite.write_graphml(G, osp.join(OUT,'Graphs',f'{dmode}_transitions_{ff:.2f}_{nn:d}.graphml'))
    nx.readwrite.write_graphml(H, osp.join(OUT,'Graphs',f'{dmode}_paths_{ff:.2f}_{nn:d}.graphml'))
    return

def reprogramming_network(stats_df,nn_l,ff_l,dump=False):
    '''
    ARGUMENTS (type) DESCRIPTION
    stats_df (pandas DataFrame) 
        columns: D_F -- final distance 
                 P_F -- % recovered
                 KNN_GP -- cell type
                 NG -- number of genes
        index levels:
                 0 -- GSM_i -- initial state
                 1 -- CTI   -- initial cell type
                 2 -- CTF   -- final cell type
    nn_l (list of int) maximum number of genes
    ff_l (list of float64) frac of reprogrammed states to call a transition
    dump (boolean) whether or not to save the graph as a graphml file
    
    RETURNS/OUTPUTS
    rpg_graph (NetworkX Digraph) Graph of the reprogramming transitions
    '''
    from sklearn.neighbors import KNeighborsRegressor as KNR
    stats_df = stats_df.set_index('NG',append=True)
    #graph_storage_d = {}
    graph_LCC_d = defaultdict(dict)
    for nn in nn_l:
        cti_ctf_frac = {}
        for (cti,ctf),grp in stats_df.groupby(level=[1,2]):
            cti_ctf_frac[(cti,ctf)] = (grp.index.get_level_values(-1)<=nn).sum()/grp.shape[0]
        cti_ctf_ser = pd.Series(cti_ctf_frac)
        for ff in ff_l:
            SEL = cti_ctf_ser[cti_ctf_ser>ff]
            G = nx.DiGraph()
            G.add_edges_from(SEL.index.tolist())
            LSCC = [len(c) for c in sorted(nx.strongly_connected_components(G), key=len, reverse=True)][0]
            LWCC = [len(c) for c in sorted(nx.weakly_connected_components(G), key=len, reverse=True)][0]
            #graph_storage_d[(nn,ff)]=G
            graph_LCC_d[(nn,ff)]['LSCC']=LSCC
            graph_LCC_d[(nn,ff)]['LWCC']=LWCC
    ## for the grid estimate the LCC as a function of nn/ff
    Xg,Yg = np.mgrid[1:10:10j, 0:1:100j]
    positions = np.vstack([Xg.ravel(),Yg.ravel()]).T
    vals = pd.DataFrame(dict(graph_LCC_d)).T.LSCC
    clf = KNR(6,weights='distance')
    clf.fit(vals.index.tolist(),vals.values)
    Z = clf.predict(positions)
    Z = Z.reshape(Xg.shape)
    plt.rcParams['svg.fonttype'] = 'none'
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #IMG = ax.pcolormesh(Z, cmap='viridis')
    ax.scatter(vals.index.get_level_values(0),
               vals.index.get_level_values(1),
               s=2*vals.values,marker='o',color='#7f7f7f')
    if dmode=='RNASeq':
        ax2 = fig.add_axes([.95,1,.1,.95])
        scale = np.array([1,6,12,18,24,30,36])
        if not osp.exists(osp.join(OUT,'fig4')):
            os.makedirs(osp.join(OUT,'fig4'))
        opfn = osp.join(OUT,'fig4',f'{dmode}_transitions_landscape.svg')
    else:
        ax2 = fig.add_axes([.95,.1,.1,.85])
        scale = np.array([1,9,18,27,36,45,54])
        if not osp.exists(osp.join(OUT,'fig4')):
            os.makedirs(osp.join(OUT,'fig4'))
        opfn = osp.join(OUT,'fig4',f'{dmode}_transitions_landscape.svg')
    ax2.scatter(np.zeros(7),np.arange(7),s=2*scale,
               marker='o',color='#7f7f7f')
    ax2.axis('off')
    #cb = fig.colorbar(IMG)
    #cb.ax.set_label('Largest component size')
    ax.set_xlabel('Number of perturbed genes',size=8)
    ax.set_ylabel('Reprogramming success threshold',size=8)
    fig.savefig(opfn)
    ## save selected graph
    #annotate_network(stats_df,7,.33)
    return 0

def canc_trends(stats_df,ch):
    ## dmode=='GeneExp'
    micounts = pd.read_csv(osp.join(IN,dmode,'cell_type_phenotype.csv'),index_col=0)
    ng_counts_d = defaultdict(list)
    for (cti,ctf),grp in stats_df.groupby(level=[1,2]):
        if micounts.loc[cti,'isCancer'] and micounts.loc[ctf,'isCancer']:
            kw = 'C2C'
        elif micounts.loc[cti,'isCancer'] and not micounts.loc[ctf,'isCancer']:
            kw = 'C2N'
        elif not micounts.loc[cti,'isCancer'] and micounts.loc[ctf,'isCancer']:
            kw = 'N2C'
        elif not micounts.loc[cti,'isCancer'] and not micounts.loc[ctf,'isCancer']:
            kw = 'N2N'
        ng_counts_d[kw].extend(grp.NG.values.tolist())
    ng_counts_d = dict(ng_counts_d)
    percentage_d = {}
    for k,v in ng_counts_d.items():
        VCS = pd.Categorical(v).value_counts()
        VCS /= VCS.sum()
        SEL = VCS.iloc[:4].copy()
        SEL.index = [str(ind) for ind in SEL.index.tolist()]
        SEL['5+']=1-SEL.sum()
        
        percentage_d[k] = SEL
    perc_df = pd.DataFrame(percentage_d)
    perc_df.to_pickle(osp.join(OUT,dmode,f'{ch}_pct_df.pkl'))
    return

def plot_pies_C():
    if not osp.exists(osp.join(OUT,f'{dmode}_E_pct_df.pkl')):
        E_stats = pd.read_pickle(osp.join(OUT,dmode,'E_stats_df.pkl'))
        canc_trends(E_stats,'E')
    pie_df = pd.read_pickle(osp.join(OUT,dmode,'E_pct_df.pkl'))
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(3,2),dpi=300)
    
    alg_ax_d = dict(zip(['C2C','C2N','N2C','N2N'],[ax1,ax2,ax3,ax4]))
    alg_const_d = {'C2C':r'Cancer$\rightarrow$Cancer',
                   'C2N':r'Cancer$\rightarrow$Normal',
                   'N2C':r'Normal$\rightarrow$Cancer',
                   'N2N':r'Normal$\rightarrow$Normal'}
    al_d={'5+':0.1,'1':0.9,'2':0.7,'3':0.5,'4':0.3}
    gn_l = ['1','2','3','4','5+']
    for alg,ax in alg_ax_d.items():
        const = alg_const_d[alg]
        dat = pie_df.loc[gn_l,alg]
        #frac = dat.loc[gn_l]/dat.sum()
        #pct_dist= np.asarray([ 1.25 if f<.09 else .6 for f in frac])
        wdgs,lbls,pcts = ax.pie(dat,colors=['C0'],
            autopct='%1.1f%%',shadow=False,pctdistance=1.2,startangle=90)
        #[lbl.set_family('serif') for lbl in lbls]
        #[lbl.set_size(6) for lbl in lbls]
        #[pct.set_family('serif') for pct in pcts]
        [pct.set_size(6) for pct in pcts]
        [wdg.set_alpha(al_d[gn_l[ii]]) for ii,wdg in enumerate(wdgs)]
        ax.set_xlabel(const,size=10)
    if not osp.exists(osp.join(OUT,'fig7')):
        os.makedirs(osp.join(OUT,'fig7'))
    fig.savefig(osp.join(OUT,'fig7','pie_charts_C.svg'))
    return

def plot_pies(dmode):
    E_stats = pd.read_pickle(osp.join(OUT,dmode,'E_stats_df.pkl'))
    A_stats = pd.read_pickle(osp.join(OUT,dmode,'A_stats_df.pkl'))
    I_stats = pd.read_pickle(osp.join(OUT,dmode,'I_stats_df.pkl'))

    E_NG = E_stats.loc['NG'].value_counts(normalize=True).sort_index()
    I_NG = I_stats.loc['NG'].value_counts(normalize=True).sort_index()
    A_NG = A_stats.loc['NG'].value_counts(normalize=True).sort_index()

    percentage_d = {}
    for nm,df in zip(['E','I','A'],[E_NG,I_NG,A_NG]):
        SEL = df.iloc[:4].copy()        
        SEL.index = [str(ind) for ind in SEL.index.tolist()]
        SEL['5+']=1-SEL.sum()
        percentage_d[nm] = SEL
    pct_df = pd.DataFrame(percentage_d).fillna(0)
    fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(4,1),dpi=300)
    
    alg_clr_d = {'A':'#d62728','I':'#2ca02c','E':'#1f77B4'}
    alg_ax_d = dict(zip(['A','I','E'],[ax1,ax2,ax3]))
    alg_const_d = dict(zip(['A','I','E'],['None','Size','Size & Sign']))
    al_d={'5+':0.1,'1':.9,'2':.7,'3':.5,'4':.3}
    gn_l = ['1','2','3','4','5+']
    for alg,ax in alg_ax_d.items():
        dat = pct_df.loc[gn_l,alg]
        const = alg_const_d[alg]
        wdgs,lbls,pcts = ax.pie(dat,colors=[alg_clr_d[alg]],
            autopct='%1.1f%%',shadow=False,pctdistance=1.2,startangle=90)
        [pct.set_size(6) for pct in pcts]
        [wdg.set_alpha(al_d[gn_l[ii]]) for ii,wdg in enumerate(wdgs)]
        ax.set_xlabel(const,size=10)
    if not osp.exists(osp.join(OUT,'fig7')):
        os.makedirs(osp.join(OUT,'fig7'))
    fig.savefig(osp.join(OUT,'fig7',f'{dmode}_constraint_comparison_piecharts.svg'))
    return

def gen_nums(k_v,N,pv=[]):
    ## termination conditions
    if N==0:
        ## no more successes to report
        return pv+[0 for dum in range(len(k_v))]
    elif sum(k_v)==N:
        ## all remaining trials must be successful
        return pv+[kk for kk in k_v]
    elif len(k_v)==1:
        # only one remaining category
        return pv+[N]
    ## loop over the total number of possible numbers
    ## and call the function with one less element
    mv = max(0,N+k_v[0]-sum(k_v))
    MV = min(N,k_v[0])+1
    OPL = []
    for n0 in range(mv,MV):
        OPL.extend(gen_nums(k_v[1:],N-n0,pv+[n0]))
    return OPL ## will need to reshape this 

def calc_log_ncr(n,k,q):
    mv = min(n,k-n)
    MV = max(n,k-n)
    rv = n*np.log(q)
    if mv==0:
        return rv
    else:
        rv += np.sum(np.log(range(MV+1,k+1)))-np.sum(np.log(range(1,mv+1)))
        return rv

def approx_probs(probs):
    pit = [(k,v) for k,v in probs.items()]
    L = len(pit)
    Z = list(zip(pit[:-1],pit[1:]))
    ii = 0
    cond_probs = []
    while ii < (L-1):
        (x1,y1),(x2,y2)=Z[ii]
        if np.abs(x1-x2) < .02: # == 1/53.
            cond_probs.append(((x1+x2)/2.,y1+y2))
            ii+=2
        else:
            cond_probs.append((x1,y1))
            ii+=1

    if ii==(L-1):
        cond_probs.append((x2,y2))
    return pd.Series(dict(cond_probs))
    
def PMF(N,k_v,p_v):
    p_v = np.array(p_v)
#     if not np.dot(k_v,p_v)==1:
#         print('Probability is not properly normalized:',np.dot(k_v,p_v))
#         return -1
    q_v = p_v/(1-p_v)
    PF = np.exp(np.sum(np.array(k_v)*np.log(1-p_v)))
    prb_l = []
    #for NN in range(N):
    arr = np.array(gen_nums(k_v,N))
    if arr.shape[0]!=len(k_v):
        dat_l =[]
        arr = arr.reshape(arr.shape[0]//len(k_v),len(k_v))
        for jj in range(arr.shape[0]):
            dat_l.append(np.exp(sum([calc_log_ncr(arr[jj,ii],k_v[ii],q_v[ii]) for ii in range(len(k_v))])))
        S = sum(dat_l)
    else:
        S = np.exp(sum([calc_log_ncr(arr[ii],k_v[ii],q_v[ii]) for ii in range(len(k_v))]))
    if np.isnan(S) or np.isinf(S):
        import pdb
        pdb.set_trace()
    return PF*S

def calc_probs(max_gn_df,all_prob,other_freq_ser,useCols=True,NN=54):
    ct_d = {}
    if useCols:
        L = list(max_gn_df.gn_max.unstack().items())
    else:
        L = list(max_gn_df.gn_max.unstack().iterrows())
    for ct,col in L:
        print(ct)
        vcs = col.value_counts()
        for gn,val in vcs.items():
            ct_d[(ct,gn,'binom')] = binom(NN,all_prob.loc[(gn,)]).logsf(val)
            probs = []
            for ser in other_freq_ser.values:
                if dmode=='RNASeq':
                    ser.index=pd.MultiIndex.from_tuples(ser.index.tolist())
                prb = ser.loc[gn] if gn in ser.index else 0
                probs.append(prb)
            probs = pd.Categorical(np.array(probs)).value_counts().sort_index(ascending=True)
            if np.allclose(probs.index[0],0):
                Nzero = probs.iloc[0]
                UB = NN-Nzero
                probs = probs.iloc[1:]
            else:
                UB = NN
#            if probs.shape[0]>10:
#                probs = approx_probs(probs)
            K_V = probs.values
            P_V = probs.index.tolist()
            pPMF = partial(PMF,k_v=K_V,p_v=P_V)
            if val >= UB/2:
                ct_d[(ct,gn,'multinom')] = sum(list(futures.map(pPMF,range(val,UB))))
            else: # val < UB/2:
                ct_d[(ct,gn,'multinom')] = 1-sum(list(futures.map(pPMF,range(val))))
    return pd.Series(ct_d)

def calc_gn_freqs(dmode):
    E_gnfreq = pd.read_pickle(osp.join(OUT,dmode,'E_gnfreq_df.pkl'))
    NN= 36 if dmode=='RNASeq' else 54
    ctp_ngn_d = defaultdict(dict)
    for ctp, grp in E_gnfreq.T.groupby(level=[1,2]):
        mu = grp.mean()
        ctp_ngn_d['val_max'][ctp] = mu.max()
        ctp_ngn_d['gn_max'][ctp] = mu.idxmax()
        ctp_ngn_d['n_exp'][ctp] = grp.shape[0]
    max_gn_df = pd.DataFrame(dict(ctp_ngn_d))

    cti_freq_d = {}
    ctf_freq_d = {}
    max_gn_mat =  max_gn_df.gn_max.unstack()
    for cti,row in max_gn_mat.iterrows():
        cti_freq_d[cti] = row.value_counts(normalize=True)
    cti_freq_ser = pd.Series(cti_freq_d)

    for ctf,col in max_gn_mat.items():
        ctf_freq_d[ctf] = col.value_counts(normalize=True)
    ctf_freq_ser = pd.Series(ctf_freq_d)
    all_prob = max_gn_df.gn_max.value_counts(normalize=True)
    if dmode=='RNASeq':
        all_prob.index = pd.MultiIndex.from_tuples(all_prob.index.tolist())                        
    finct_d = calc_probs(max_gn_df,all_prob,cti_freq_ser,useCols=True,NN=NN)
    initct_d = calc_probs(max_gn_df,all_prob,ctf_freq_ser,useCols=False,NN=NN)
    init_pvals = 1-np.abs(2*np.exp(pd.Series(initct_d))-1)
    finct_pvals = 1-np.abs(2*np.exp(pd.Series(finct_d))-1)
    DF = pd.Series(finct_d).unstack()
    DF.loc[:,'binom'] =  1-np.abs(2*np.exp(DF.binom)-1)
    DF.loc[:,'multinom'] = 1-np.abs(2*DF.multinom-1)

    N_multi = DF[multipletests(DF.multinom,0.01,method='fdr_tsbh')[0]].shape[0]
    N_binom = DF[multipletests(DF.binom,0.01,method='fdr_tsbh')[0]].shape[0]

    DF.to_pickle(osp.join(OUT,dmode,'E_finct_gn_pvals.pkl'))
    DF2 = pd.Series(initct_d).unstack()
    DF2.loc[:,'binom'] =  1-np.abs(2*np.exp(DF2.binom)-1)
    DF2.loc[:,'multinom'] = 1-np.abs(2*DF2.multinom-1)

    N_multi_init = DF2[multipletests(DF2.multinom,0.01,method='fdr_tsbh')[0]].shape[0]
    N_binom_init = DF2[multipletests(DF2.binom,0.01,method='fdr_tsbh')[0]].shape[0]

    DF2.to_pickle(osp.join(OUT,dmode,'E_initct_gn_pvals_old.pkl'))
    return DF,DF2    

def write_network(DF,DF2):
    multi_init = DF2[multipletests(DF2.multinom,0.01,method='fdr_tsbh')[0]]
    multi_fin =  DF[multipletests(DF.multinom,0.01,method='fdr_tsbh')[0]]
    if dmode=='GeneExp':
        mipheno = pd.read_csv(osp.join(IN,dmode,'cell_type_phenotype.csv'),index_col=0)
    else:
        mipheno = pd.read_csv(osp.join(IN,dmode,'celltype_color_tissue.csv'),index_col=0)
    G = nx.DiGraph()
    edge_pairs = []
    gn_list = []
    ct_list = []
    for (cti,gn),pval in multi_init.multinom.items():
        edge_pairs.append((cti,gn,-1*np.log10(np.clip(pval,1e-20,1))))
        gn_list.append(gn)
        ct_list.append(cti)

    for (ctf,gn),pval in multi_fin.multinom.items():
        edge_pairs.append((gn,ctf,-1*np.log10(np.clip(pval,1e-20,1))))
        gn_list.append(gn)
        ct_list.append(ctf)
    G.add_weighted_edges_from(edge_pairs)
    for nd in G.nodes():
        if nd in mipheno.index:
            if dmode=='GeneExp':
                G.nodes[nd]['name']=str(nd.replace(';','\n').replace('_','\n'))
                if mipheno.loc[nd,'isCancer']:
                    G.nodes[nd]['class']='cancer'
                else:
                    G.nodes[nd]['class']='normal'
            else:
                G.nodes[nd]['name']=str(nd)
                G.nodes[nd]['class']='cell_type'
                G.nodes[nd]['tissue_group']=str(mipheno.loc[nd,'NAME'])
                G.nodes[nd]['color']=mipheno.loc[nd,'COLOR']
        elif nd in gn_list:
            if dmode=='GeneExp':
                ND = nd.replace('_kd','-').replace('_oe','+')
                if '_' in ND:
                    ND = ND.split('_',1)[1]
            else:
                ND = nd
            G.nodes[nd]['name']=str(ND)
            G.nodes[nd]['class']='gene'
    if not osp.exists(osp.join(OUT,'Graphs')):
        os.makedirs(osp.join(OUT,'Graphs'))            
    nx.readwrite.write_graphml(G,osp.join(OUT,'Graphs',f'{dmode}_significant_genes_old.graphml'))
    return 0

def main(stats_df):
    if dmode == 'RNASeq':
        annotate_network(stats_df,5,.5)
        nn_l = range(1,11)
        ff_l = np.linspace(.05,.95,19)
        reprogramming_network(stats_df,nn_l,ff_l)
        DF,DF2 = calc_gn_freqs(dmode)
        write_network(DF,DF2)
        plot_pies(dmode)
        
    else:
        annotate_network(stats_df,2,.75)
        nn_l = range(1,6)
        ff_l = np.linspace(.05,.95,19)
        reprogramming_network(stats_df,nn_l,ff_l)
        DF,DF2 = calc_gn_freqs(dmode)
        write_network(DF,DF2)
        plot_pies_C()
        plot_pies(dmode)
        

if __name__ == '__main__':
    for ch in ['E']:
        stats_df = pd.read_pickle(osp.join(OUT,dmode,f'{ch}_stats_df.pkl')).T
        main(stats_df)
