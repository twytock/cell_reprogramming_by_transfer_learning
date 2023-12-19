#!/usr/bin/env python
# encoding: utf-8
"""
naive_v_opt_final.py

Created by Thomas Wytock on 2023-11-01.
"""

import time,gc,sys,os
from itertools import permutations as perms
from glob import glob
import os.path as osp, pickle as P, numpy as np
import matplotlib as mpl, matplotlib.pyplot as plt
import pandas as pd
from functools import partial
from operator import itemgetter as ig
from collections import defaultdict
from scoop import futures
import cplex


index_gsym = pd.read_table('data/GeneExp/gpl570_entrezg_rowmap.txt',index_col=0) # inclulded
deltas = pd.read_csv('data/GeneExp/fig2_sel_deltas.csv.gz',index_col=[0,1])
colkw_gn_ser = pd.read_csv('data/GeneExp/fig2_sel_pert2gn_ser.csv',index_col=0).iloc[:,0]

## need to select data as in forward selection

with open('data/GeneExp/fig2_ct_list.txt','r') as fh:
    cts = [ln.strip() for ln in fh]

def cplex_optimize_const_parallel(Q,cvec,**kwargs):
    '''
    Perform constrained optimization with CPLEX.

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
    soln_l : list of lists (of length 1)
        optimal values of the control inputs
    lpstat_l : list
        statuses of the cplex solver

    '''
    Q = 0.5*(Q+Q.T)
    Q = np.atleast_2d(Q) if not isinstance(Q,np.ndarray) else Q
    numvars = Q.shape[0];
    Q = Q + np.eye(numvars)*1e-12
    #cvec = cvec[:,0]
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
        c.variables.add(obj=cvec[:,0].tolist(),ub=UB,lb=LB)
    elif 'lb' in kwargs.keys():
        LB = kwargs['lb']
        LB = LB if isinstance(LB,np.array) else np.ones(numvars)*LB
        LB = LB.tolist()
        c.variables.add(obj=cvec[:,0].tolist(),lb=LB)
    elif 'ub' in kwargs.keys():
        UB = kwargs['ub']
        UB = UB if isinstance(UB,np.array) else np.ones(numvars)*UB
        UB = UB.tolist()
        c.variables.add(obj=cvec[:,0].tolist(),ub=UB)
    else:
        c.variables.add(obj=cvec[:,0].tolist())
    c.objective.set_sense(c.objective.sense.minimize)
    c.objective.set_quadratic(qmat)
    soln_l = []
    lpstat_l = []
    for ii in range(cvec.shape[1]-1):
        c.solve()
        lpstat = c.solution.get_status()
        lpstat_l.append(lpstat)
        x = c.solution.get_values()
        soln_l.append(x)
        c.objective.set_linear(enumerate(cvec[:,ii+1]))
    c.solve()
    lpstat = c.solution.get_status()
    lpstat_l.append(lpstat)
    x = c.solution.get_values()
    soln_l.append(x)    
    return soln_l,lpstat_l

def cplex_optimize_numconst_parallel(Q,cvec,**kwargs):
    '''
    Perform a series of constrained optimizations with CPLEX.
    This variant returns a list of solutions that have a single nonzero gene.

    Parameters
    ----------
    Q : 2-D, square, numpy array
        Quadratic component of the objective function
    cvec : numpy array
        Linear component of the objective function; same length as Q
    kwargs : dict of optional arguments
        lb : lower bounds for the variables
        ub : upper bounds for the variables
        NG : number of nonzero variables
        
    Returns
    -------
    soln_l : list of lists
        optimal values of the control inputs
    lpstat_l : list
        statuses of the cplex solver

    '''
    Q = 0.5*(Q+Q.T)
    Q = np.atleast_2d(Q) if not isinstance(Q,np.ndarray) else Q
    numvars = Q.shape[0];
    Q = Q + np.eye(numvars)*1e-12
    #cvec = cvec[:,0]
    qmat = []
    for ii in range(2*numvars):
        if ii < numvars:
            qmat.append([['x%d'%(xx+1) for xx in range(numvars)],Q[ii,:].tolist()])
        else:
            qmat.append([['z%d' % (ii-numvars+1) ],[1e-12]])    
    c = cplex.Cplex()
    out = c.set_results_stream(None)
    out = c.set_log_stream(None)
    out = c.set_error_stream(None)
    out = c.set_warning_stream(None)
    c.parameters.optimalitytarget.set(c.parameters.optimalitytarget.values.optimal_global)
    varnames = ['x%d' % (ii+1) for ii in range(numvars)]
    indvarnames = ['z%d' % (ii+1) for ii in range(numvars)]
    if 'lb' in kwargs.keys() and 'ub' in kwargs.keys():
        LB = kwargs['lb']
        LB = LB if isinstance(LB,np.ndarray) else np.ones(numvars)*LB
        LB = LB.tolist()
        UB = kwargs['ub']
        UB = UB if isinstance(UB,np.ndarray) else np.ones(numvars)*UB
        UB = UB.tolist()
        c.variables.add(obj=cvec[:,0].tolist(),ub=UB,lb=LB)
    elif 'lb' in kwargs.keys():
        LB = kwargs['lb']
        LB = LB if isinstance(LB,np.array) else np.ones(numvars)*LB
        LB = LB.tolist()
        c.variables.add(obj=cvec[:,0].tolist(),lb=LB)
    elif 'ub' in kwargs.keys():
        UB = kwargs['ub']
        UB = UB if isinstance(UB,np.array) else np.ones(numvars)*UB
        UB = UB.tolist()
        c.variables.add(obj=cvec[:,0].tolist(),ub=UB)
    else:
        c.variables.add(obj=cvec[:,0].tolist())
    c.variables.add(types=[c.variables.type.binary]*numvars ,names=indvarnames)
    for ii in range(numvars):
        c.indicator_constraints.add(indvar='z%d' % (ii+1),
                                    complemented=1,rhs=0,sense='E',
                                    lin_expr=cplex.SparsePair(ind=['x%d'%(ii+1)],val=[1]),
                                    name='ind%d' % (ii+1))
    c.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=indvarnames,val=np.ones(numvars))],
                             rhs=[kwargs['NG']],senses=['E'],names=['lconst'])
    c.objective.set_sense(c.objective.sense.minimize)
    c.objective.set_quadratic(qmat)
    soln_l = []
    lpstat_l = []
    for ii in range(cvec.shape[1]-1):
        c.solve()
        lpstat = c.solution.get_status()
        lpstat_l.append(lpstat)
        x = c.solution.get_values()
        soln_l.append(x[:numvars])
        c.objective.set_linear(enumerate(cvec[:,ii+1]))
    c.solve()
    lpstat = c.solution.get_status()
    lpstat_l.append(lpstat)
    x = c.solution.get_values()
    soln_l.append(x[:numvars])    
    return soln_l,lpstat_l

def updated_recw_cplex_fs_parallel(S,dcols,T,lb,ub,opt,nGene=-1):
    '''
    Convert initial states, selected columns, and different optimization parameters
    into the inputs appropriate for calling Cplex.
    
    Parameters
    ----------
    S : pandas Series, initial state gene expression vector
    dcols : list or pandas Index, names of the perturbations
    T : pandas Series, target state gene expression vector
    lb : float, lower bounds for the variables
    ub : float, upper bounds for the variables
    opt : Boolean, whether to calculate the optimal solution (True) or naive solution (False)
    nGene : integer, number of genes in the solution (-1 for all)
    
    Returns
    -------
    al_df : pandas DataFrame, contains the amounts that each gene needs to be perturbed
    d1 : pandas Series, final distance to the target for the optimal perturbation
    pf : pandas Series, percentage of distance to the target recovered for the optimal perturbation
    
    '''
    if not opt:
        if isinstance(dcols,list):
            dcols = dcols + [kw]
        else:
            dcols = dcols.tolist()+ [kw] # add kw to index
    d = deltas.loc[:,dcols] 
    wsep = (S-T)
    if isinstance(d,pd.DataFrame):
        wD = d
        wC = (wsep@wD).T
        wQ = wD.T@wD
    else:
        wD = d
        wC = wsep@wD
        wQ = wD@wD
    if nGene<0:
        al_l,sc = cplex_optimize_const_parallel(wQ.values,wC.values,lb=lb,ub=ub)
    else:
        al_l,sc = cplex_optimize_numconst_parallel(wQ.values,wC.values,lb=lb,ub=ub,NG=nGene)
    al_df = pd.DataFrame(np.vstack(al_l),index=S.index,columns=dcols)
    d0 = np.sqrt(np.sum(np.square(wsep),axis=1))#np.linalg.norm(wsep,axis=1)
    d1 = np.sqrt(np.sum(np.square(wsep+al_df@(d.T)),axis=1))
    pf = 1-d1/d0
    return al_df.T,d1,pf

def updated_recw_calc_fs_parallel(S,dcols,T,opt,ch,nGene):
    '''
    Convert initial states, selected columns, and different optimization parameters
    into the inputs appropriate for analytical optimization (via linear algebra).
    
    Parameters
    ----------
    S : pandas Series, initial state gene expression vector
    dcols : list or pandas Index, names of the perturbations
    T : pandas Series, target state gene expression vector
    opt : Boolean, whether to calculate the optimal solution (True) or naive solution (False)
    ch : string, character ('A', 'I', or 'E') specifying the constraints on the perturbations
    nGene : integer, number of genes in the solution (-1 for all)
    
    Returns
    -------
    opt_vals_df : pandas DataFrame, contains the amounts that each gene needs to be perturbed
    opt_vals_df.d_fin : pandas Series, final distance to the target for the optimal perturbation
    opt_vals_df.p_fin : pandas Series, percentage of distance to the target recovered for the optimal perturbation

    '''
    if not opt:
        if isinstance(dcols,list):
            dcols = dcols + [kw]
        else:
            dcols = dcols.tolist()+ [kw] # add kw to index
    wsep = (S-T)
    d0 = np.linalg.norm(wsep,axis=1)
    if nGene<0:
        d = deltas.loc[:,dcols] 
        if isinstance(d,pd.DataFrame):
            wD = d
            wC = (wsep@wD).T
            wQ = wD.T@wD
            al = np.linalg.solve(wQ.values,-1*wC.loc[wQ.index].values)
            #d1 = np.linalg.norm(WSEP+np.dot(WD,al))
            al = pd.DataFrame(al,index=wQ.index,columns=S.index)
            d1 = np.linalg.norm(wsep + (wD@al).T,axis=1)
        else:
            wD = d
            wC = wsep@wD; wQ = wD@wD
            al = np.asarray([-1 * wC.values/wQ.values])
            d1 = np.linalg.norm(wsep.add(wD@pd.Series(al,index=wQ.index)))
        
        pf = 1-d1/d0
        return al,pd.Series(d1,index=S.index),pd.Series(pf,index=S.index)
    else: ## nGene==1
        d1_by_dcol = {}
        al_by_dcol = {}
        for dcol in dcols:
            wD = deltas.loc[:,dcol]
            wC = wsep@wD; wQ = wD@wD
            al = np.asarray([-1 * wC.values/wQ])
            if ch=='A':
                pass
            elif ch=='I':
                al = np.clip(al,-1,1)
            elif ch=='E':
                al = np.clip(al,0,1)
            d1 = np.linalg.norm(wsep + np.outer(al[0,:],wD),axis=1)
            d1_by_dcol[dcol]=d1
            al_by_dcol[dcol]=al[0,:]
        d1_by_dcol_df = pd.DataFrame(d1_by_dcol,index=wsep.index)
        al_by_dcol_df = pd.DataFrame(al_by_dcol,index=wsep.index)
        d0_ser = pd.Series(d0,index=wsep.index)
        d1_min = d1_by_dcol_df.min(axis=1)
        d1_idx = d1_by_dcol_df.idxmin(axis=1)
        al_min = pd.Series(dict([((GSM,CT),row.loc[d1_idx.loc[(GSM,CT)]]) for (GSM,CT),row in al_by_dcol_df.iterrows()]))
        p_fin = 1-d1_min/d0_ser
        opt_vals_df = pd.DataFrame({'u_star':al_min, 'd_fin':d1_min,'gpid':d1_idx,
                                    'p_fin':p_fin})
        return opt_vals_df.T,opt_vals_df.d_fin,opt_vals_df.p_fin

def ctpair_naive_new(cti,ctf):
    """Calculate the naive alphas for each initial state.
     
    Parameters
    ----------
    cti : string, name of the initial cell type
    ctf : string, name of the final cell type
    
    Returns
    -------
    naive_raw : pandas DataFrame, 
    naive_rsc : pandas DataFrame, 
    These DataFrames contain the amounts that each gene needs to be perturbed, 
    the final distance, and the percent recovery. The "rsc" file rescales the
    amounts estimated by the naive method according to the measured strength of 
    the perturbation (i.e., how much the gene expression of the perturbed gene
    is observed to change).
    
    """
    gn_colkw_d = defaultdict(list)
    for colkw,gn in colkw_gn_ser.items(): 
        gn_colkw_d[gn].append(colkw)
    gn_colkw_d = dict(gn_colkw_d)
    S = pd.Series(dict((col,deltas.loc[gn,col]) for col,gn in colkw_gn_ser.items()))
    cti_df = pd.read_csv(f'data/GeneExp/fig2_cell_line_data/{cti}_downsample.csv.gz',index_col=[0,1])
    ctf_df = pd.read_csv(f'data/GeneExp/fig2_cell_line_data/{ctf}_downsample.csv.gz',index_col=[0,1])
    ctf_mu = ctf_df.mean()
    ##look at toycell for rescaling
    naive_al = (cti_df.sub(ctf_mu))
    sel=naive_al.columns.isin(gn_colkw_d.keys())
    naive_al_sel = naive_al.T[sel].T
    gn_op_d = {}
    for gn,row in naive_al_sel.items():
        #gn = index_gsym.loc[probe,'symbol.ALL']
        if gn not in gn_colkw_d.keys():
            continue
        ## loop through each column (gene) and calculate alphas
        sel_keys = gn_colkw_d[gn]
        if len(sel_keys)>1:
            for (GSM,CT),chng in row.items():
                pkw = f'1_{gn}_oe'
                nkw = f'1_{gn}_kd'
                if chng>0:
                    gn_op_d[(pkw,GSM,CT,'rsc')]=np.abs(chng/S.loc[pkw])
                    gn_op_d[(pkw,GSM,CT,'raw')]=np.abs(chng)
                    gn_op_d[(nkw,GSM,CT,'rsc')]=0
                    gn_op_d[(nkw,GSM,CT,'raw')]=0
                else:
                    gn_op_d[(nkw,GSM,CT,'rsc')]=np.abs(chng/S.loc[pkw])
                    gn_op_d[(nkw,GSM,CT,'raw')]=np.abs(chng)
                    gn_op_d[(pkw,GSM,CT,'rsc')]=0
                    gn_op_d[(pkw,GSM,CT,'raw')]=0
        else:
            Zid = gn_colkw_d[gn][0]
            mult = -1 if Zid.endswith('kd') else 1
            if Zid=='1_FGF2_kd':
                mult=1
            for (GSM,CT),chng in row.items():
                gn_op_d[(Zid,GSM,CT,'rsc')]=np.abs(chng/S.loc[Zid])
                gn_op_d[(Zid,GSM,CT,'raw')]=chng*mult
    naive_df = pd.Series(gn_op_d).unstack()
    return (naive_df.raw).unstack(0),(naive_df.rsc).unstack(0)

def ctpair_opt(cti,ctf,ch,nGene):
    """Retrieve the optimal solutions for a cell type pair.

    Parameters
    ----------
    cti : string, name of the initial cell type
    ctf : string, name of the final cell type
    ch : string, character ('A', 'I', or 'E') specifying the constraints on the perturbations
    nGene : integer, number of genes in the solution (-1 for all)
    
    Returns
    -------
    d1_df : pandas DataFrame, final distance to the target for the optimal perturbation
    pf_df : pandas DataFrame, percentage of distance to the target recovered for the optimal perturbation
    al_df : pandas DataFrame, contains the amounts that each gene needs to be perturbed

    """
    init = pd.read_csv(f'data/GeneExp/fig2_cell_line_data/{cti}_downsample.csv.gz',index_col=[0,1])
    ctf_df = pd.read_csv(f'data/GeneExp/fig2_cell_line_data/{ctf}_downsample.csv.gz',index_col=[0,1])
    targ = ctf_df.mean()    
    dcols = deltas.columns
    kw = f'{cti}-{ctf}'
    W=np.ones(targ.shape[0])
    if ch == 'A':
        if nGene<0:
            al_df,d1_df,pf_df = updated_recw_calc_fs_parallel(init,dcols,targ,True,ch,-1)
        else:
            al_df,d1_df,pf_df = updated_recw_calc_fs_parallel(init,dcols,targ,True,ch,1)
    elif ch == 'I':
        if nGene<0:
            al_df,d1_df,pf_df=updated_recw_cplex_fs_parallel(init,dcols,targ,-1,1,True,-1)
        elif nGene==1:
            al_df,d1_df,pf_df = updated_recw_calc_fs_parallel(init,dcols,targ,True,ch,1)
        else:
            ## need to redo this using the analytic solution + clipping
            al_df,d1_df,pf_df=updated_recw_cplex_fs_parallel(init,dcols,targ,-1,1,True,1)
    elif ch == 'E':
        if nGene<0:
            al_df,d1_df,pf_df=updated_recw_cplex_fs_parallel(init,dcols,targ,0,1,True,-1)
        elif nGene==1:
            al_df,d1_df,pf_df = updated_recw_calc_fs_parallel(init,dcols,targ,True,ch,1)
        else:
            al_df,d1_df,pf_df=updated_recw_cplex_fs_parallel(init,dcols,targ,0,1,True,1)
    return d1_df,pf_df,al_df.T

def calc_dists(p,cti,ctf):
    """Calculate the distances in gene expression space between initial and target states.
    
    Parameters
    ----------
    p : pandas DataFrame, transcriptional responses to perturbations for each state
    cti : string, name of the initial cell type
    ctf : string, name of the final cell type
    
    Returns
    -------
    df : pandas DataFrame, final distance to the target after applying perturbation p
    
    """
    cti_df = pd.read_csv(f'data/GeneExp/fig2_cell_line_data/{cti}_downsample.csv.gz')
    ctf_df = pd.read_csv(f'data/GeneExp/fig2_cell_line_data/{ctf}_downsample.csv.gz')
    ctf_mu = ctf_df.mean()
    if len(p.index.names)<2:
        p.index.names=['GSM']
    else:
        p.index.names=['GSM','CT']
    X = np.sqrt(np.square((cti_df.loc[p.index] + p).sub(ctf_mu)).sum(axis=1))
    Z = np.sqrt(np.square(cti_df.sub(ctf_mu)).sum(axis=1))
    X.name = 'DF_naive'; Z.name = 'DI' 
    #Y.name = 'DF_opt'; 
    df = pd.concat([X,Z], axis=1)
    df['PF_naive'] = 1-df.DF_naive.div(df.DI)
    #df['PF_opt'] = 1-df.DF_opt.div(df.DI)
    df["CTF"] = np.repeat(ctf,df.shape[0])
#     df = df.reset_index()
#     C = df.columns
#     C = [c if c != "CT" else "CTI" for c in C ]
#     df.columns = C
    return df

def qualitative_sim(al1,al2):
    """Calculate the alignment of the signs of the elements of two perturbation presciptions.
    
    Parameters
    ----------
    al1 : pandas Series, perturbation prescription for the naive method
    al2 : pandas Series, perturbation prescription for the optimal method
    
    Returns
    -------
    myser : pandas Series, sign alignment of the perturbation prescriptions
   
    """
    myd = {}
    for ind1,r1 in al1.iterrows():
        r1 = r1.loc[al2.columns]; r2 = al2.loc[ind1]
        myd[ind1] = 2*np.sum(np.sign(r1)==np.sign(r2))/float(r1.shape[0])-1
    myser = pd.Series(myd)
    return myser

def compare_freq(naive_stat,opt_stat):
    """Compre the relative frequencies of genes found from the naive and optimal methods.
    
    Parameters
    ----------
    naive_stat : pandas DataFrame, perturbation prescriptions of the genes for the naive method
    opt_stat : pandas DataFrame, perturbation prescription of the genes for the optimal method
    Columns of the DataFrames are:
    "gpid": the gene perturbation identifier
    "u_star": the perturbation prescription for a given gene
    
    Returns
    -------
    S/tot : float, the total fraction of genes aligning
    Sexp : float, the expected fraction of genes aligning
    np.sqrt(Sexp_sqr-Sexp**2) : float, the standard deviation of the fraction of genes aligning
   
    """
    naive_pos = naive_stat[naive_stat.u_star>0]
    naive_neg = naive_stat[naive_stat.u_star<0]
    opt_pos = opt_stat[opt_stat.u_star>0]
    opt_neg = opt_stat[opt_stat.u_star<0]
    tot = opt_stat.gpid.value_counts().sum()
    tot2 = naive_stat.gpid.value_counts().sum()
    opt_pf = opt_pos.gpid.value_counts()/tot
    opt_nf = opt_neg.gpid.value_counts()/tot
    naive_pf = naive_pos.gpid.value_counts()/tot2
    naive_nf = naive_neg.gpid.value_counts()/tot2
    S = 0
    for ind in opt_pos.index.intersection(naive_pos.index):
        if opt_pos.loc[ind,'gpid']==naive_pos.loc[ind,'gpid']:
            S+=1#opt_pf.loc[ind]*naive_pf.loc[ind]
    for ind in opt_pos.index.intersection(naive_neg.index):
        if opt_pos.loc[ind,'gpid']==naive_neg.loc[ind,'gpid']:
            S-=1#opt_pf.loc[ind]*naive_nf.loc[ind]        
    for ind in opt_neg.index.intersection(naive_pos.index):
        if opt_neg.loc[ind,'gpid']==naive_pos.loc[ind,'gpid']:
            S-=1#opt_nf.loc[ind]*naive_pf.loc[ind]
    for ind in opt_neg.index.intersection(naive_neg.index):
        if opt_neg.loc[ind,'gpid']==naive_neg.loc[ind,'gpid']:
            S+=1#opt_nf.loc[ind]*naive_nf.loc[ind]
    Sexp=0
    Sexp_sqr = 0
    for ind in opt_pf.index.intersection(naive_pf.index):
        Sexp+=opt_pf.loc[ind]*naive_pf.loc[ind]
        Sexp_sqr+=(opt_pf.loc[ind]*naive_pf.loc[ind])
    for ind in opt_pf.index.intersection(naive_nf.index):
        Sexp-=opt_pf.loc[ind]*naive_nf.loc[ind]
        Sexp_sqr+=(opt_pf.loc[ind]*naive_nf.loc[ind])
    for ind in opt_nf.index.intersection(naive_pf.index):
        Sexp-=opt_nf.loc[ind]*naive_pf.loc[ind]
        Sexp_sqr+=(opt_nf.loc[ind]*naive_pf.loc[ind])
    for ind in opt_nf.index.intersection(naive_nf.index):
        Sexp+=opt_nf.loc[ind]*naive_nf.loc[ind]
        Sexp_sqr+=(opt_nf.loc[ind]*naive_nf.loc[ind])
    return S/tot,Sexp,np.sqrt(Sexp_sqr-Sexp**2)

def naive_opt_single_gene(naive_u,cti,ctf):
    """
    Parameters
    ----------
    naive_u : pandas DataFrame, perturbation prescriptions of the genes for the naive method
    cti : string, name of the initial cell type
    ctf : string, name of the final cell type
    
    Returns
    -------
    opt_vals_df : pandas DataFrame, statistics for the genes selected and differences in the naive method
    
"""
    cti_df = pd.read_csv(f'data/GeneExp/fig2_cell_line_data/{cti}_downsample.csv.gz',index_col=[0,1])
    ctf_df = pd.read_csv(f'data/GeneExp/fig2_cell_line_data/{ctf}_downsample.csv.gz',index_col=[0,1])
    ctf_mu = ctf_df.mean()
    if len(naive_u.index.names)<2:
        naive_u.index.names=['GSM']
    else:
        naive_u.index.names=['GSM','CT']
    opt_vals_d = defaultdict(dict)
    diff_df = cti_df.sub(ctf_mu)
    for (GSM,CT),diff in diff_df.iterrows():
        d0 = np.linalg.norm(diff)
        XO_MAT = np.atleast_2d(np.ones(naive_u.shape[1])).T@np.atleast_2d(diff)
        PERT = deltas.loc[:,naive_u.columns] *naive_u.loc[(GSM,CT)]
        d1s = pd.Series(np.linalg.norm(XO_MAT+PERT.T,axis=1),index=naive_u.columns)
        opt_vals_d[(GSM,CT)]['gpid'] = d1s.idxmin()
        opt_vals_d[(GSM,CT)]['d_fin'] = d1s.min()
        opt_vals_d[(GSM,CT)]['p_fin'] = 1-d1s.min()/d0
        opt_vals_d[(GSM,CT)]['u_star'] = np.sign(naive_u.loc[(GSM,CT),d1s.idxmin()])
    opt_vals_df = pd.DataFrame(dict(opt_vals_d)).T
    return opt_vals_df


def rescue(cti,ctf,ch,nGene):
    """Calculate the naive and optimal results for all (cti,ctf) pairs.
    
    
    """        
    if nGene < 0:
        OUT_FN = f'output/naive_v_opt/{ch}/{cti}-{ctf}.pkl'
    elif nGene ==1:
        OUT_FN = f'output/naive_v_opt/{ch}/{cti}-{ctf}_{nGene:d}.pkl'
        if osp.exists(OUT_FN):
            print(OUT_FN,'already completed.')
            return 1
    naive_al,naive_rsc = ctpair_naive_new(cti,ctf)
    d1,pf,opt_al = ctpair_opt(cti,ctf,ch,nGene)
    if ch == 'I':
        naive_al = naive_al.clip(-1,1)
    elif ch == 'E':
        naive_al = naive_al.clip(0,1)
    if nGene<0:
        pp = naive_al@(deltas.T) #(deltas@naive_al.T).T
        df = calc_dists(pp,cti,ctf)
        SER= qualitative_sim(naive_al,opt_al)
        df.loc[SER.index,'L1_align'] = SER
        df = df.set_index('CTF',append=True)
        df['DF_opt']=d1
        df['PF_opt']=pf        
        df.to_pickle(OUT_FN)
    elif nGene==1:
        opt_sg_vals_raw = naive_opt_single_gene(naive_al,cti,ctf)
        L1_align,L1_exp,L1_std = compare_freq(opt_sg_vals_raw,opt_al) 
        L1_align_ser = pd.Series(dict([(kk,L1_align) for kk in opt_sg_vals_raw.index]))
        L1_exp_ser = pd.Series(dict([(kk,L1_exp) for kk in opt_sg_vals_raw.index]))
        L1_std_ser = pd.Series(dict([(kk,L1_std) for kk in opt_sg_vals_raw.index]))
        op_df = pd.DataFrame({'DF_opt':d1,'PF_opt':pf,'DF_naive':opt_sg_vals_raw.d_fin,
                              'PF_naive':opt_sg_vals_raw.p_fin,'L1_align':L1_align_ser,
                              'L1_exp':L1_exp_ser,'L1_std':L1_std_ser})
        op_df.to_pickle(OUT_FN)
                
    return 1


def plot_naive_v_opt_recovery(oneGN=False):
    """Makes the panels of Fig. 2

    Parameters
    ----------
    oneGN : Boolean, determines whether the figure is generated for single-gene solutions (True) or all-gene solutions (False)
    
    Returns
    -------
    
    None
    
    The function generates an .svg file in the directory "output/fig2/"

    """
    plt.rcParams['svg.fonttype'] = 'none'
    myD = {}
    for CH in ['A','I','E']:
        print(CH)
        for ctf in cts:
            #print('---> %s' % ctf)
            fn_l = glob(f'output/naive_v_opt/{CH}/*-{ctf}.pkl') if not oneGN else glob(f'output/naive_v_opt/{CH}/*-{ctf}_1.pkl')
            for fn in fn_l:
                CTI = fn.split('/')[-1].split('-')[0]
                X = pd.read_pickle(fn)
                myD[(CH,ctf,CTI)] = X.loc[:,['PF_naive','PF_opt','L1_align']].median()
    data_df = pd.DataFrame(myD).T

    label2abbr={'acute_lymphoblastic_T_cell_leukaemia':'ALTL',
                'acute_myeloid_leukaemia':'AML',
                'blast_phase_chronic_myeloid_leukaemia':'BP-CML',
                'breast;carcinoma':'BC',
                'breast;ductal_carcinoma':'BDC',
                'clear_cell_renal_cell_carcinoma':'CRC',
                'diffuse_large_B_cell_lymphoma':'DLBL',
                'endometrium;adenocarcinoma':'EA',
                'glioblastoma':'GBM',
                'glioma':'GMA',
                'hepatocellular_carcinoma':'HC',
                'large_intestine;adenocarcinoma':'LIA',
                'large_intestine;carcinoma':'LIC',
                'lung;adenocarcinoma':'LUA',
                'lung;large_cell_carcinoma':'LULC',
                'lung;non_small_cell_carcinoma':'LUNSC',
                'lung;small_cell_carcinoma':'LUSC',
                'lung;squamous_cell_carcinoma':'LUSQ',
                'malignant_melanoma':'MM',
                'ovary;adenocarcinoma':'OA',
                'ovary;carcinoma':'OC',
                'pancreas;ductal_carcinoma':'PDC',
                'plasma_cell_myeloma':'PCM',
                'prostate;adenocarcinoma':'PRA',
                'prostate;carcinoma':'PRC',
                'renal_cell_carcinoma':'RC',
                'stomach;adenocarcinoma':'STA',
                'activated_M2_macrophage':'AM2M',
                'activated_myeloid_dendritic_cell':'AMD',
                'adipose_tissue_subcutaneous':'ASC',
                'alveolar_macrophage':'ALM',
                'alveolar_macrophage_from_smoker':'ALM-S',
                'colon':'C',
                'naive_CD4+_T_cell':'CD4+T',
                'coronary_smooth_muscle_cell':'CSM',
                'cultured_airway_epithelial_cell':'CAE',
                'ductal_breast_epithelial_cell':'DBE',
                'embryonic stem cells':'ES',
                'foreskin_fibroblast':'FF',
                'hepatocyte':'HC',
                'immortalized_airway_epithelial_cell':'IAE',
                'induced_pluripotent_cell':'IP',
                'kidney':'KID',
                'liver_tissue':'LIV',
                'mesenchymal_stem_cell':'MCS',
                'monocyte':'MC',
                'myeloid_dendritic_cell':'MD',
                'M2_macrophage':'M2M',
                'nasal_epithelial_cell':'NE',
                'neutrophil':'NP',
                'small_airway_epithelial_cell':'SAE',
                'skin_cultured_fibroblast':'SCF',
                'skeletal_muscle':'SKM',
                'vein_endothelial_cell':'VE'}
    ch_clr_d = {'A':'C3','I':'C2','E':'C0'}
    alg_alph_d = {'PF_naive':0.5,'PF_opt':.9,'L1_align':.9}
    fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(3.33,9),sharey=True)
    ax3.axvline(0,color='k')
    s = 0.05
    SF = False if not oneGN else True
    w = 0.25
    ch_offset_d = {'A':0,'I':w+s,'E':2*w+2*s}
    for CH,grp in data_df.groupby(level=0):
        sgrp = grp.copy()
        sgrp.index = sgrp.index.droplevel(0)
        for kw,col in sgrp.items():
            srt_data = sorted([IT for IT in col.groupby(level=0)],key=lambda IT: IT[0].lower())
            ytls,data_l = zip(*srt_data)
            if kw=='PF_naive':
                boxes1 = ax1.boxplot(data_l,positions=np.arange(len(ytls))+ch_offset_d[CH]+s,widths=w,
                            vert=False, patch_artist=True, showfliers=SF,
                            boxprops={'color':ch_clr_d[CH],'alpha':alg_alph_d[kw]},
                            whiskerprops={'color':ch_clr_d[CH]},capprops={'color':ch_clr_d[CH]},
                            medianprops={'color':'C7'},flierprops={'alpha':0.5,'markeredgecolor':None})
                for elt in boxes1['boxes']:
                    elt.set_color(ch_clr_d[CH])
                    elt.set_alpha(alg_alph_d[kw])                    
            elif kw=='PF_opt':
                boxes2 = ax2.boxplot(data_l,positions=np.arange(len(ytls))+ch_offset_d[CH]+s,widths=w,
                            vert=False, patch_artist=True, showfliers=SF,
                            boxprops={'color':ch_clr_d[CH],'alpha':alg_alph_d[kw]},
                            whiskerprops={'color':ch_clr_d[CH]},capprops={'color':ch_clr_d[CH]},
                            medianprops={'color':'C7'},flierprops={'alpha':0.5,'markeredgecolor':None})                
                for elt in boxes2['boxes']:
                    elt.set_color(ch_clr_d[CH])
                    elt.set_alpha(alg_alph_d[kw])                    
            else:
                boxes3 = ax3.boxplot(data_l,positions=np.arange(len(ytls))+ch_offset_d[CH]+s,widths=w,
                            vert=False, patch_artist=True, showfliers=SF,
                            boxprops={'color':ch_clr_d[CH],'alpha':0.9},
                            whiskerprops={'color':ch_clr_d[CH]},capprops={'color':ch_clr_d[CH]},
                            medianprops={'color':'C7'},flierprops={'alpha':0.5,'markeredgecolor':None})
                for elt in boxes3['boxes']:
                    elt.set_color(ch_clr_d[CH])
                    elt.set_alpha(alg_alph_d[kw])                    
    
    ax1.set_xlabel(r'$R^2$',size=8)
    ax2.set_xlabel(r'$R^2$',size=8)
    ax1.set_yticks(np.arange(len(ytls))+w+s)
    yabbrs=[label2abbr[ytl] for ytl in ytls]
    ax1.set_yticklabels(yabbrs,size=6)
    plt.setp(ax1.get_xticklabels(),rotation=0,size=6)
    plt.setp(ax2.get_xticklabels(),rotation=0,size=6)
    plt.setp(ax3.get_xticklabels(),rotation=0,size=6)    
    ax1.set_ylim(-0.25,len(ytls)+0.25)
    ax3.set_xlabel(r'$R^2_{\mathrm{sign}}$',size=8)
    if oneGN:
        if not osp.exists('output/figS1/'):
            os.makedirs('output/figS1/')        
        fig.savefig('output/figS1/Naive_v_Opt_median_new_1.svg')
    else:
        if not osp.exists('output/fig2/'):
            os.makedirs('output/fig2/')
        fig.savefig('output/fig2/Naive_v_Opt_median_new.svg')
    return


def main():
    ## generate results for each constraint condition and number of genes
    t_i = time.time()
    for ng in [1,-1]:
        for CH in ['A','I','E']:
            if not osp.exists(f'output/naive_v_opt/{ch}/'):
                os.makedirs(f'output/naive_v_opt/{ch}/')
            resc_red = partial(rescue,ch=CH,nGene=ng)
            all_list = [elt for elt in zip(*perms(cts,2))] ## can limit the length of "elt" to speed computation
            L = list(futures.map(resc_red,*all_list)) ## this step will generate parallel processes
        ## generate figure
        if ng<0:
            plot_naive_v_opt_recovery(oneGN=False)
        else:
            plot_naive_v_opt_recovery(oneGN=True)
    t_f = time.time()
    print(f'Elapsed time: {t_f-t_i:.02f} seconds')

if __name__ == '__main__':
    main()
