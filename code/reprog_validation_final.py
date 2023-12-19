#!/usr/bin/env python
# encoding: utf-8
"""
reprog_validation_final.py

Created by Thomas Wytock on 2023-11-02
"""

import numpy as np
import pandas as pd
import sys,os,os.path as osp
from scoop import futures
from sklearn.neighbors import KNeighborsClassifier as KNC
from functools import partial
from itertools import combinations as icombs, product as iprod
from operator import itemgetter as ig
from collections import defaultdict
from glob import glob

CH = sys.argv[1]
if CH in ('E','I'):
    import cplex


n_neighbors = 8
dmode = 'GeneExp'
IN = f'data/{dmode}'
OUT = 'output'

annot = pd.read_excel(f'{IN}/ReprogrammingExperiments.xlsx', index_col = 0, header=0)
reprog_gse_arr = annot.GSE.unique()

annot2 = pd.read_excel(f'{IN}/all_genexp_data.xlsx' ,index_col=0)
corr_data_bc = pd.read_csv(f'{IN}/nonseq_bc_corr_data_all.csv',dtype=str).set_index(['Unnamed: 0','Unnamed: 1']).astype(float)
FEAT_FN = 'tissue_feat_l_WC_all.txt'
with open(osp.join(f'{IN}',f'{FEAT_FN}'),'r') as fh:
    feat_l = [ln.strip() for ln in fh]
    if dmode=='GeneExp':
        feat_l = feat_l[:4]


corr_data_bc = corr_data_bc.loc[:,feat_l]
unpert = pd.read_csv(f'{IN}/unpert_ct_inds.csv',index_col=0)
unpert = pd.MultiIndex.from_frame(unpert).droplevel(0)
SEL_UNPERT = corr_data_bc.index.get_level_values(0).isin(unpert.get_level_values(0))
mod_data1 = corr_data_bc[SEL_UNPERT]

def weights(df):
    """
    Calculate the weights for each eigengene. 
    Weights satisfy \sum_i w^2_i = 1 (sum(WT**2)=1).

    Parameters
    ----------
    df : pandas DataFrame, DataFrame of transcriptional data projected onto eigengenes (columns)
        
    Returns
    -------
    WT : pandas Series, weights associated with each eigengene
    """
    WT = df.std(axis=0,ddof=1).apply(lambda i: 1/i)
    if np.any(np.isinf(WT)):
        M = np.amax(WT[~np.isinf(WT)])
        WT[np.isinf(WT)]=2*M
    WT = WT.div(np.linalg.norm(WT.astype(np.float64)))
    return WT

def fix_indices(mod_data):
    '''
    Change the cell type so all tissues in the human body index are included

    Parameters
    ----------
    mod_data : pandas DataFrame
        DataFrame of transcriptional data whose index will be modified
        
    Returns
    -------
    mod_data : pandas DataFrame
        DataFrame with modified cell types in level 1 of the MultiIndex

    '''
    L = []
    ## loop over indicies
    for gsm,ct in mod_data.index.tolist():
        if ct in ['fetal_day100_150_lung', 'fetal_day_75_100_lung']:
            L.append((gsm,'fetal_lung_fibroblasts'))
        elif gsm=='GSM689064':
            L.append((gsm,'in_vitro_differentiated_fibroblast'))
        else:
            L.append((gsm,ct))
    mod_data.index = pd.MultiIndex.from_tuples(L)
    mod_data.index.names = ['GSM','CT']
    return mod_data
mod_data1 = fix_indices(mod_data1)
corr_data_bc = fix_indices(corr_data_bc)
## list of "wild-type" cell types
LL = ['None','used in the balanced background','No','mock','healthy','DMSO',
      'wt','nc','naive embryonic stem cells','mock; 5d', 'mock 1uM 3d','BMP2+',
      'mock; 12h','mock 21d','mock 10h','DMSO 24h','DMSO 6h','DMSO 2h','TGFb3+',
      'DMSO 4h','DMSO; 72h','DMSO 1h','DMSO; 4h','IL-3','EPO; 7d','EPO; 14d',
      'DMSO 18h','DMSO 72h','DMSO 10h','DMSO 48h','DMSO 21d','TGFb; 0h',
      'passage 19', 'passage 13','human-induced-NPC']
## select unperturbed cell types.
othergsms = annot2[annot2.Treatment.isin(LL)&(annot2.GeneTarget=='None')].index.tolist()
## select reprogrammed cell types.
othergsms2 = annot2[(annot2.CellType=='induced_pluripotent_cell')&(annot2.Treatment=='oe')].index.tolist()
othergsms = list(set(othergsms)|set(othergsms2)) + ['GSM689064']
## limit to only the unperturbed and reprogrammed cell types
otherInds = corr_data_bc[corr_data_bc.index.get_level_values(0).isin(othergsms)].index
combInds = set(otherInds.tolist()) | set(mod_data1.index.tolist())
selMI = pd.MultiIndex.from_tuples(combInds)
selMI.names=['GSM','CT']
mod_data = corr_data_bc.loc[selMI]
VCS = mod_data.index.get_level_values(1).value_counts()
gt10cts = VCS[VCS>=10].index.tolist()
tissue_df = pd.read_csv(f'{IN}/matched_tissues_for_validation.csv')
tissue_d = dict(zip(tissue_df.iloc[:,0],tissue_df.iloc[:,1]))
SEL = mod_data.index.get_level_values(1).isin(set(tissue_d.values())|set(gt10cts))
ovrlap_tissues = set(mod_data.index.get_level_values(1)) & set(tissue_d.values())
mod_data=mod_data[SEL]
CAT = pd.Categorical(mod_data.index.get_level_values(level='CT'))
idx = CAT.categories
Y = CAT.codes
WT = weights(mod_data)    
scaled_states =mod_data.multiply(WT)
clf = KNC(n_neighbors, weights='distance')
clf.fit(scaled_states.values,Y)
#all_scaled_data = corr_data_bc.multiply(WT)

def calc_reprog_deltas():
    '''
    Calculate the transcriptional responses to reprogramming interventions.
    This script relies on global variables (annot2 and annot).
    
    Returns
    -------
    delta_df : pandas DataFrame, matrix of transcriptional responses, rows = genes, columns = perturbations

    '''    
    DELTA_d = {}
    for gse in reprog_gse_arr:
        grp = annot2[annot2.GSE==gse].copy()
        grp.loc[grp.index,'Treatment2'] = annot.loc[grp.index,'Treatment2'].copy()
        grp = grp.fillna('NA')
        TRT_arr = grp.Treatment.unique()
        SSS =set(['oe','nc']) & set(TRT_arr)
        if not len(SSS)>1:
            print('%s missing keys: %s.' % (gse,SSS))
            continue
        mu_d = {}
        for (CT,GN,trt1,trt2),ggrp in grp.groupby(['CellLineName','GeneTarget', 'Treatment','Treatment2']):
            trtkw = '%s,%s' % (trt1,trt2) if trt2 != 'NA' else trt1
            SEL = corr_data_bc.index.get_level_values(0).isin(ggrp.index)
            mu_d[(gse,CT,GN,trtkw)] = corr_data_bc[SEL].mean(axis=0)
        for (k1,CT,GN,k2),v in mu_d.items():
            if not k2.startswith(('nc','wt')):
                if ',' in k2:
                    trt2 = k2.split(',')[1]
                    cc_kw = 'nc,%s' % (trt2)
                    ctrlkw = cc_kw if (k1,CT,'None',cc_kw) in mu_d.keys() else 'nc'
                    nckey = (k1,CT,'None',ctrlkw)
                    if not nckey in mu_d.keys():
                        if gse=='GSE23583' and CT == 'dH1CF':
                            nckey = ('GSE23583', 'dH1F', 'None', 'nc')
                else:
                    ctrlkw = 'nc'
                    nckey = (k1,CT,'None',ctrlkw)
                    if not nckey in mu_d.keys():
                        if gse=='GSE23583' and CT == 'dH1CF':
                            nckey = ('GSE23583', 'CVCL-3653', 'None', 'nc')
                        elif gse=='GSE33536' and CT in ('cardiac fibroblasts','PBMC'):
                            nckey = ('GSE33536', 'skin_epithelial_cell_keratinocyte', 'None', 'nc')
                dkw = ';'.join([k1,CT,GN,k2])
                DELTA_d[dkw] = v-mu_d[nckey]
    delta_df = pd.DataFrame(DELTA_d)
    delta_df.to_csv(f'{IN}/deltas_reprog_nci60.csv')
    return delta_df

if osp.exists(f'{IN}/deltas_reprog_nci60.csv'):
    reprog_deltas = pd.read_csv(f'{IN}/deltas_reprog_nci60.csv',dtype=str).set_index('Unnamed: 0').astype(float)
else:
    reprog_deltas = calc_reprog_deltas()
reprog_deltas = reprog_deltas.loc[feat_l]
non_rpg_delt = pd.read_csv(f'{IN}/delta_corr_newnci60-2.csv.gz',dtype=str).set_index('Unnamed: 0').astype(float)
non_rpg_delt = non_rpg_delt.loc[feat_l]
comb_deltas = pd.concat([non_rpg_delt,reprog_deltas],sort=False,axis=1)


def cplex_optimize_const(Q,cvec,**kwargs):
    '''
    Perform constrained optimization with CPLEX

    Parameters
    ----------
    Q : 2-D, square, numpy array
        Quadratic component of the objective function
    cvec : numpy array
        Linear component of the objective function; same length as Q
    kwargs : dict of optional arguments
        lb : lower bounds for the variables
        ub : upper bounds for the variables
        
    Returns
    -------
    x : list
        optimal values of the control inputs
    lpstat : int
        status of the cplex solver

    '''
    Q = 0.5*(Q+Q.T)
    Q = np.atleast_2d(Q) if not isinstance(Q,np.ndarray) else Q
    numvars = Q.shape[0];
    Q = Q + np.eye(numvars)*1e-12
    qmat = []
    for ii in range(numvars):
        qmat.append([range(numvars),Q[ii,:].tolist()])
    c = cplex.Cplex()
    out = c.set_results_stream(None)
    out = c.set_log_stream(None)
    out = c.set_error_stream(None)
    out = c.set_warning_stream(None)
    c.parameters.optimalitytarget.set(c.parameters.optimalitytarget.values.optimal_global)
    if 'lb' in kwargs.keys() and 'ub' in kwargs.keys():
        LB = kwargs['lb']
        LB = LB if isinstance(LB,np.ndarray) else np.ones(numvars)*LB
        LB = LB.tolist()
        UB = kwargs['ub']
        UB = UB if isinstance(UB,np.ndarray) else np.ones(numvars)*UB
        UB = UB.tolist()
        c.variables.add(obj=cvec.tolist(),ub=UB,lb=LB)
    elif 'lb' in kwargs.keys():
        LB = kwargs['lb']
        LB = LB if isinstance(LB,np.array) else np.ones(numvars)*LB
        LB = LB.tolist()
        c.variables.add(obj=cvec.tolist(),lb=LB)
    elif 'ub' in kwargs.keys():
        UB = kwargs['ub']
        UB = UB if isinstance(UB,np.array) else np.ones(numvars)*UB
        UB = UB.tolist()
        c.variables.add(obj=cvec.tolist(),ub=UB)
    else:
        c.variables.add(obj=cvec.tolist())        
    c.objective.set_sense(c.objective.sense.minimize)
    c.objective.set_quadratic(qmat)
    c.solve()
    lpstat = c.solution.get_status()
    x = c.solution.get_values()
    return x,lpstat

def updated_recw_cplex_fs(kw,dcols,S,T,deltas,W,lb,ub,opt):
    '''
    Perform constrained forward selection

    Parameters
    ----------
    kw : string
        keyword of the perturbation to be added.
    dcols : list
        columns of the perturbation matrix that are fixed.
    S : pandas Series
        vector of INITIAL gene expression state.
    T : pandas Series
        vector of TARGET gene expression state.
    deltas : pandas DataFrame
        perturbation matrix; rows = gene responses, columns = perturbations.
    lb : int or numpy array
        lower bounds of the variables; if type==int, lb is the same for all variables
    ub : int or numpy array
        upper bounds of the variables; if type==int, ub is the same for all variables
    W : pandas Series
        weights vector to determine how each feature applies to data.
    opt : Boolean
        whether or not to calculate the optimal perturbation.
        if opt is True, only the optimal perturbation including all genes will be performed
        otherwise, forward selection will proceed fixing the columns in "selcols" and adding
        columns one at a time

    Returns
    -------
    kw : string
        keyword of the perturbation to be added.
    al : pandas Series
        vector of the extent to which each perturbation is on (!=0)/off (==0).
    d1 : float
        final distance of the target state after perturbation.
    pf : float
        percentage of the initial distance recovered as a result of the perturbation.
    '''    
    if not opt:
        if isinstance(dcols,list):
            dcols = dcols + [kw]
        else:
            dcols = dcols.tolist()+ [kw] # add kw to index
    d = deltas.loc[:,dcols]
    if np.any(np.isinf(W)):
        M=np.amax(W[~np.isinf(W)])
    wsep = S.sub(T).multiply(W)
    d0 = np.linalg.norm(wsep)
    if isinstance(d,pd.DataFrame):
        wD = d.T.multiply(W).T
        wC = wsep.dot(wD); wQ = wD.T.dot(wD)
    elif isinstance(d,pd.Series):
        wD = d.T.multiply(W).T
        wC = wsep.dot(wD)
        wQ = wD.T.dot(wD)
    al,sc = cplex_optimize_const(wQ.values,wC.values,lb=lb,ub=ub)
    if isinstance(d,pd.DataFrame):
        d1 = np.linalg.norm(wsep.add(wD.dot(pd.Series(al,index=wD.columns))))
    elif isinstance(d,pd.Series):
        d1 = np.linalg.norm(wsep.add(wD.dot(pd.Series(al,index=wD.name))))
    pf = 1-d1/d0
    return kw,al,d1,pf


def test_reprog(gse):
    '''
    main function to test the reprogramming potential of the library of perturbations

    Parameters
    ----------
    gse : string, label of the GEO series accession
        
    Returns
    -------
    0
        results are saved in files.

    comb_deltas : pandas DataFrame, (global variable)
        perturbations to be tested (columns) and genetic responses (rows).

    '''
    
    grp = annot2[annot2.GSE==gse].copy()
    grp.loc[grp.index,'Treatment2'] = annot.loc[grp.index,'Treatment2'].copy()
    grp = grp.fillna('None')
    print(gse)
    TRT_arr = grp.Treatment.unique()
    SSS =set(['wt','nc']) & set(TRT_arr)
    if not len(SSS)>1:
        print(f'{gse} missing keys: {SSS}.')
        return 0
    ## 'nc' -> 'wt'
    ## gather the data associated with each cell line transition
    data_d = {}
    for (CLN,GN,trt1,trt2),ggrp in grp.groupby(['CellType','GeneTarget','Treatment','Treatment2']):
        if ggrp.CellType.value_counts().shape[0]>1:
            print('multiple cell types',gse,CLN)
        CT = ggrp.CellType.unique()[0]
        if trt1 == 'nc':
            trtkw = f'nc,{CT}' if trt2 != 'NA' else f'nc,{CT},{trt2}'
            ## may need to double check that the CT still matches what was fit (in cases that it was)                
        elif trt1 == 'wt':
            trtkw = f'wt,{CT}' if trt2 != 'NA' else f'wt,{CT},{trt2}' 
        elif trt1 == 'oe':
            trtkw = 'oe,iPS' if trt2 != 'NA' else f'oe,iPS,{trt2}'
        SEL = corr_data_bc.index.get_level_values(0).isin(ggrp.index)
        if SEL.sum() <1:
            print('Missing Data.')
            continue
        data_d[trtkw] = corr_data_bc[SEL].loc[:,feat_l]
    
    ## for each pair of keywords, calculate the optimal perturbations
    for K1,K2 in icombs(data_d.keys(),2):
        if not K1.startswith('nc') and not K2.startswith('nc'):
            continue
#         elif osp.exists('%s/%s/FS/%s_%s-%s_%s.pkl' % (PRJ,OUT,gse,K1,K2,ch)):
#             continue
        if K1.startswith('nc'):
            S = data_d[K1]
            T = data_d[K2]
        else:
            S = data_d[K2]
            T = data_d[K1]
        #T should be correct now
        SRF = T.index.get_level_values(1)
        CTF = SRF.value_counts().index[0]
        t_basin=idx.get_indexer([CTF])[0]
        i_basin=idx.get_indexer(S.index.get_level_values(1)[:1])[0]
        if t_basin<0:
            t_basin = clf.predict(T.iloc[:1,:]*WT)[0]
            CTF = CAT[t_basin]
        part_fs = partial(forward_selection, sel_cols=[], 
                          deltas=comb_deltas, model=clf, WT=WT, 
                          ctf=CTF, tbasin=t_basin, ibasin=i_basin,N=1)
        ## loop over S/T and find the best single perturbation 
        ## and the best optimal perturbation
        opl = [part_fs(S_T) for S_T in  iprod(S.iterrows(),T.iterrows())]
        #gsm_res_df_l= list(futures.map(part_FS,cti_df.iterrows()))
        #if INTERMED:
        #    gsm_res_df_l.extend(comp_gr_df_l)
        df_l = []
        for gsmi,gsmf,df in opl:
            #df['GSM_i']=[gsm for _ in range(df.shape[0])]
            df_l.append(df)
        #    if comp_df.shape[0] > 1:
        #        df_l.append(comp_df)
        DF = pd.concat(df_l)
        if not osp.exists(osp.join(OUT,'FS_GeneExp')):
            os.makedirs(osp.join(OUT,'FS_GeneExp'))
        
        DF.to_pickle(osp.join(OUT,'FS_GeneExp',f'{gse}_{K1}-{K2}_{CH}.pkl'))
    return 0

        
def updated_reduce_fs(res,dcols,opt,Swt,dwt,model,ctigrp,ctfgrp):
    '''
    gathers forward selection data into a vector of results and the oselected columns

    Parameters
    ----------
    res : tuple of 2 elements
        column (1st element) and amount of optimal perturbations (2nd element).
    dcols : list
        selected perturbation columns.
    opt : Boolean
        whether or not this is an optimal perturbation including all columns.
    Swt : pandas Series
        weighted intial state.
    dwt : pandas DataFrame
        DESCRIPTION.
    model : sci-kit learn KNN classifier
        KNN model that predicts cell type based on gene expression.

    Returns
    -------
    tuple
        dat_ser -- the optimal perturbation, distance recovery, and reprogramming statistics
        sc -- the selected columns.

    '''
    # instead of re-fitting the model, pass it as a parameter
    # S is simply the initial state from the scaled data
    # deltas are the perturbations scaled by the weights
    al_ser= pd.Series(np.zeros(dwt.shape[1]),index=dwt.columns)
    if not opt:
        if isinstance(dcols,list):
            sc = dcols+[res[0]]
        else:
            sc = dcols.tolist() + [res[0]] # add column
    else:
        sc = dcols
    if len(sc)>0:
        al_ser.loc[sc]=res[1]
        d = dwt.loc[:,sc]
        R = Swt.add(d.dot(al_ser.loc[sc]))
    else:
        R=Swt
    C = model.predict(R.values.reshape(1,-1))[0]
    P,I,F = model.predict_proba(R.values.reshape(1,-1))[0,[C,ctigrp,ctfgrp]]
    dat_ser=pd.Series(list(res[2:])+[C,P,I,F],
                      index=['D_F','P_F','KNN_GP','C_PRB','I_PRB','F_PRB'])
    dat_ser = pd.concat([dat_ser,al_ser])
    return dat_ser,np.asarray(sc,dtype='object')
        
        
def forward_selection(S_T,sel_cols,deltas,model,WT,ctf,ibasin,tbasin,IM=False,N=-1):
    '''
    performs forward selection of perturbations to steer between cell types

    Parameters
    ----------
    S_T : tuple of 2 elements
        (1st element) tuple of ((sample accession, cell type), initial state).
            sample accession and cell type are strings; initial state is a pandas Series
        (2nd element) same types as first element, but for the target state
    sel_cols : list
        selected perturbation columns.
    deltas : pandas DataFrame
        trancriptional responses to perturbations.
    Swt : pandas Series
        weighted intial state.
    dwt : pandas DataFrame
        DESCRIPTION.
    model : sci-kit learn KNN classifier
        KNN model that predicts cell type based on gene expression.

    Returns
    -------
    tuple
        dat_ser -- the optimal perturbation, distance recovery, and reprogramming statistics
        sc -- the selected columns.

    '''
    ((gsmi,clni),S),((gsmf,clnp),T) = S_T
    p_add = partial(updated_add_1_col,init=S,targ=T,deltas=deltas,W=WT)
    p_red = partial(updated_reduce_fs,Swt=S.multiply(WT),dwt=deltas.T.multiply(WT).T,
                    model=model,ctigrp=ibasin,ctfgrp=tbasin)
    red_ser_d = {}
    ## test for improper basins
    if not model.predict(T.multiply(WT).values.reshape(1,-1))[0]==tbasin:
        print("Invalid Target",T.name)
        #find point closest to the center to aim at
        T = mod_data.groupby(level='CT').first().loc[ctf]
        p_add = partial(updated_add_1_col,init=S,targ=T,deltas=deltas,W=WT)
    if model.predict(S.multiply(WT).values.reshape(1,-1))[0]==\
       model.predict(T.multiply(WT).values.reshape(1,-1))[0]:
        print("Overlap",gsmi,gsmf)
        res = ['none',
               pd.Series(np.zeros(deltas.shape[1]),index=deltas.columns),
               np.linalg.norm(S.sub(T).multiply(WT)),0]
        red_ser, sel_cols = p_red(res,[],True)
        red_ser_d[-1]=red_ser
        DF = pd.DataFrame(red_ser_d).T
        DF.index.name='NG'; DF=DF.reset_index()
        return S.name,T.name,DF
    if CH == 'E':
        if N != 1:
            opt_res = updated_recw_cplex_fs('opt_al',deltas.columns,S,T,deltas,WT,0,1,True)
            opt_red,opt_sc = updated_reduce_fs(opt_res,deltas.columns,True,S.multiply(WT),
                                               deltas.T.multiply(WT).T,model,
                                               ctigrp=ibasin,ctfgrp=tbasin)   
            NGMAX = deltas.shape[1] - np.sum(opt_res[1]==0)
            NGrange = range(len(sel_cols),NGMAX)
        else:
            NGrange = range(N)
    else: 
        NGrange = range(len(sel_cols),len(deltas.columns)) if N<0 else range(N)
    for zz,NG in enumerate(NGrange):
        opt=True if len(sel_cols) == len(deltas.columns) else False
        res_l =p_add(sel_cols)
        if N!=1:
            res = sorted(res_l,key=ig(2))[0]
            red_ser, sel_cols = p_red(res,sel_cols,opt)
            red_ser_d[NG+1] = red_ser
            if CH == 'E':
                red_ser_d[len(deltas.columns)] = opt_red
            if red_ser.KNN_GP == tbasin[0]:
                break
        else:
            L = []
            for kw,al,d1,pf in res_l:
                dat_ser,___ = p_red(('',al,d1,pf),[kw],True)
                L.append({'pert':kw,'alpha':al[0],'d1':d1,
                          'pf':pf,'GSMi':gsmi,'GSMf':gsmf,
                          'C_PRB':dat_ser.loc['C_PRB'],
                          'I_PRB':dat_ser.loc['I_PRB'],
                          'F_PRB':dat_ser.loc['F_PRB'],
                          'KNN_GP':dat_ser.loc['KNN_GP']})
            DF = pd.DataFrame(L)
    if N!=1:
        DF = pd.DataFrame(red_ser_d).T
        DF.index.name='NG'; DF=DF.reset_index()
    cti = corr_data_bc.xs(gsmi,level='GSM').index[0]
    return S.name,T.name,DF    

def updated_recw_calc_fs(kw,dcols,S,T,deltas,W,opt):
    '''
    performs unconstrained forward selection

    Parameters
    ----------
    kw : string
        keyword of the perturbation to be added.
    dcols : list
        columns of the perturbation matrix that are fixed.
    S : pandas Series
        vector of INITIAL gene expression state.
    T : pandas Series
        vector of TARGET gene expression state.
    deltas : pandas DataFrame
        perturbation matrix; rows = gene responses, columns = perturbations.
    W : pandas Series
        weights vector to determine how each feature applies to data.
    opt : Boolean
        whether or not to calculate the optimal perturbation.
        if opt is True, only the optimal perturbation including all genes will be performed
        otherwise, forward selection will proceed fixing the columns in "selcols" and adding
        columns one at a time

    Returns
    -------
    kw : string
        keyword of the perturbation to be added.
    al : pandas Series
        vector of the extent to which each perturbation is on (!=0)/off (==0).
    d1 : float
        final distance of the target state after perturbation.
    pf : float
        percentage of the initial distance recovered as a result of the perturbation.
    '''
    if not opt:
        if isinstance(dcols,list):
            dcols = dcols + [kw]
        else:
            dcols = dcols.tolist()+ [kw] # add kw to index
    d = deltas.loc[:,dcols]# I = S.index; dd = deltas.loc[I,dcols].values; WW = W.loc[I].values
    wsep = S.sub(T).multiply(W)# WSEP =(S.loc[I].values-T.loc[I].values)*WW
    d0 = np.linalg.norm(wsep)
    if isinstance(d,pd.DataFrame):
        wD = d.T.multiply(W).T # WD = (dd.T*WW).T
        wC = wsep.dot(wD) # WC = np.dot(WSEP,WD)
        wQ = wD.T.dot(wD) # WQ = np.dot(WD.T,WD)
        #al = -1*np.linalg.solve(WQ,WC)
        al = -1*np.linalg.solve(wQ.values,wC.loc[wQ.index].values) ## missing a factor of -1
        #d1 = np.linalg.norm(WSEP+np.dot(WD,al))
        d1 = np.linalg.norm(wsep.add(wD.dot(pd.Series(al,index=wQ.index))))
    else:
        wD = d.multiply(W) # WD = dd*W
        wC = wsep.dot(wD); wQ = wD.dot(wD) # WC = np.dot(WSEP,WD)
        #WQ = np.dot(WD.T,WD)
        al = -1*np.asarray([ wC.values/wQ.values]) # al = -1*WC/WQ
        #d1 = np.linalg.norm(WSEP+np.dot(WD,al))
        d1 = np.linalg.norm(wsep.add(wD.dot(pd.Series(al,index=wQ.index))))
    pf = 1-d1/d0
    return kw,al,d1,pf


def updated_add_1_col(selcols,init,targ,deltas,W,opt=False):
    '''
    adds the single gene to the perturbation set that most improves reprogramming

    Parameters
    ----------
    selcols : list
        list of column identifiers for the genetic perturbations.
    init : pandas Series
        expression values of the initial state.
    targ : pandas Series
        expression values of the final state.
    deltas : pandas DataFrame
        perturbation matrix; rows = gene responses, columns = perturbations.
    W : pandas Series
        weights vector to determine how each feature applies to data.
    opt : boolean, optional
        whether or not to calculate the optimal perturbation. The default is False.
        if opt is True, only the optimal perturbation including all genes will be performed
        otherwise, forward selection will proceed fixing the columns in "selcols" and adding
        columns one at a time
    CH : string (global variable)
        character that determines which constraint class to use.
        A -- unconstrained
        I -- u \in [-1, 1]
        E -- u \in [ 0, 1]

    Returns
    -------
    results : list
        list of distance recoveries and the associated perturbations.

    '''
    #uses W(weights) instead of tsigs(variances) (more convenient for this application)
    #init and targ should be UNWEIGHTED inputs
    addcols = list(set(deltas.columns)-set(selcols))
    if CH in ['I','E']:
        #print "CPLEX selected"
        if CH=='E':
            p_opt= partial(updated_recw_cplex_fs,dcols=selcols,S=init,T=targ,
            deltas=deltas,W=W,lb=0,ub=1,opt=False)
        elif CH=='I':
            p_opt= partial(updated_recw_cplex_fs,dcols=selcols,S=init,T=targ,
            deltas=deltas,W=W,lb=-1,ub=1,opt=False)
    elif CH in ['A']:
        #print "Linear Algebra selected"
        p_opt=partial(updated_recw_calc_fs,dcols=selcols,S=init,T=targ,
                      deltas=deltas,W=W,opt=False)
    results = map(p_opt,addcols)
    #results = list(futures.map(p_opt,addcols))
    return results

def main():
    #reprog_gse_arr
    opl = list(futures.map(test_reprog,reprog_gse_arr))
    return 0

def main2(CTI_CTF,tmpdir):
    CTI,CTF = CTI_CTF
    OPFN = osp.join(OUT,tmpdir,f'{CTI}-{CTF}.pkl') 
    if osp.exists(OPFN) or CTI==CTF:
        return 0
    if not osp.exists(osp.join(OUT,tmpdir)):
        os.makedirs(osp.join(OUT,tmpdir))
    print(CTI,CTF) 
    t_basin=idx.get_indexer([CTF])[0]
    i_basin=idx.get_indexer([CTI])[0]
    part_fs = partial(forward_selection, sel_cols=[], 
                  deltas=comb_deltas, model=clf, WT=WT, ch=sys.argv[1], 
                  ctf=CTF, ibasin=i_basin, tbasin=t_basin,N=1)
    S = mod_data.xs(CTI,level=1,drop_level=False).loc[:,feat_l]
    T = mod_data.xs(CTF,level=1,drop_level=False).loc[:,feat_l]
    opl = list(map(part_fs,iprod(S.iterrows(),T.iterrows())))
    comb_df = pd.concat([e[2] for e in opl],axis=0,sort=False)
    comb_df = comb_df.set_index(['GSMi','GSMf','pert'])
    comb_df.to_pickle(OPFN)
    return 0

def clean_one_gene_valid():
    inp_dir = osp.join(OUT,'one_gene_valid',CH)
    for fn in glob(osp.join(inp_dir,'*.pkl')):
        ctp = fn.split('/')[-1].split('.pkl')[0]
        CTI,CTF = ctp.split('-')
        t_basin=idx.get_indexer([CTF])[0]
        i_basin=idx.get_indexer([CTI])[0]
        test = pd.read_pickle(fn)
        S_l = []
        for col in ['alpha','d1','pf','I_PRB','F_PRB','C_PRB']:
            S = test[col].unstack().describe().loc[['mean','std']].T
            S.columns = [f'{col}_mu', f'{col}_sig']
            S_l.append(S)
        gene_pert_stats = pd.concat(S_l,axis=1)
        gene_pert_stats.to_pickle(osp.join(inp_dir,f'{ctp}_pert_stats.pkl'))
        gene_finct_stats = defaultdict(dict)
        ii=0
        for pert,grp in test.KNN_GP.groupby(level='pert'):
            for nm,prb in grp.value_counts().iteritems():
                gene_finct_stats[ii]['pert']=pert
                gene_finct_stats[ii]['KNN_GP']=int(nm)
                gene_finct_stats[ii]['KNN_NUM']=prb
                gene_finct_stats[ii]['TARG']=t_basin
                gene_finct_stats[ii]['INIT']=i_basin
                ii+=1
        gene_finct_stats_df = pd.DataFrame(dict(gene_finct_stats)).T
        gene_finct_stats_df.to_pickle(osp.join(inp_dir,f'{ctp}_finct_stats.pkl'))
        os.remove(fn)
    return 0 

if __name__ == '__main__':
    ## this does the REPROGRAMMING COMPARISON -- need to uncomment
    main()
    ## this does the TISSUE COMPARISON
    if CH=='A':
        tmpdir = osp.join('one_gene_valid',CH)
        part_main = partial(main2,tmpdir=tmpdir)
        opl = list(futures.map(part_main,iprod(ovrlap_tissues,ovrlap_tissues)))
        clean_one_gene_valid()


